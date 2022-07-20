package rl.chapter12

import java.time.LocalDateTime
import java.util.Locale

import breeze.linalg._
import breeze.numerics._
import breeze.plot._
import com.typesafe.scalalogging.Logger
import rl.ApproximateDynamicProgramming.ValueFunctionApproximation
import rl._
import rl.utils.Utils.getLogisticFunction
import rl.utils._

class OptimalExerciseRL(
  val spotPrice: Double,
  val payoff: (Double, Double) => Double,
  val expiry: Double,
  val rate: Double,
  val vol: Double,
  val numSteps: Int,
  val spotPriceFrac: Double
) {
  
  val dt: Double = expiry / numSteps
  val sqrtDt: Double = sqrt(dt)
  
  def europeanPutPrice(
    strike: Double,
    timeStep: Int = 0,
    spotPrice: Double = this.spotPrice
  ): Double = BlackScholesUtils.europeanOptionPrice(
    spotPrice = spotPrice,
    strike = strike,
    expiry = expiry - timeStep * dt,
    vol = vol,
    rate = rate,
    isCall = false
  )
  
  def exerciseCurve(
    step: Int,
    prices: Iterable[Double]
  ): DenseVector[Double] = {
    val exerciseValue = prices.map(payoff(step, _))
    DenseVector(exerciseValue.toArray)
  }
  
  def optimalValueCurve(
    step: Int,
    valueFunction: ValueFunctionApproximation[WithTime[Double]],
    prices: Iterable[Double]
  ): DenseVector[Double] = {
    val exerciseValue = prices.map(p => NonTerminal(WithTime(p, step)))
    valueFunction.evaluate(exerciseValue)
  }
  
  def linearFunctionApproximation(
    featureFunctions: Seq[NonTerminal[WithTime[Double]] => Double],
    regularizationCoefficient: Double = 0.0
  ): LinearFunctionApproximation[NonTerminal[WithTime[Double]]] = {
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      regularizationCoefficient = regularizationCoefficient,
      adamGradient = OptimalExerciseRL.defaultAdamGradient
    )
  }
  
  def lspiValueFunctionAndPolicy(
    numTransitions: Int = 5.0e4.toInt,
    numIterations: Int = 100,
    featureFunctions: Seq[NonTerminal[WithTime[Double]] => Double],
    alpha: Double = 0.10
  ): (LinearFunctionApproximation[NonTerminal[WithTime[Double]]], DeterministicPolicy[WithTime[Double], Boolean]) = {
    val transitions = this.transitions(alpha).flatten.take(numTransitions)
    val initialTargetPolicy: DeterministicPolicy[WithTime[Double], Boolean] = (_: WithTime[Double]) => false
    
    val qValueFunctions: Iterator[LinearFunctionApproximation[(NonTerminal[WithTime[Double]], Boolean)]] =
      TemporalDifference.leastSquaresPolicyIteration[WithTime[Double], Boolean](
        transitions = transitions,
        featureFunctions = processFeatureFunction(featureFunctions),
        actions = (_: NonTerminal[WithTime[Double]]) => Seq(true, false),
        initialTargetPolicy = initialTargetPolicy,
        gamma = exp(-rate * dt),
        epsilon = 1.0e-5
      )
    
    val finalQValue: LinearFunctionApproximation[(NonTerminal[WithTime[Double]], Boolean)] =
      qValueFunctions.drop(numIterations).next()
    getValueFunctionAndPolicyFromLeastSquaresQValueFunction(finalQValue)
  }
  
  def processFeatureFunction(
    featureFunctions: Seq[NonTerminal[WithTime[Double]] => Double]
  ): Seq[((NonTerminal[WithTime[Double]], Boolean)) => Double] = {
    def functor(f: NonTerminal[WithTime[Double]] => Double)
      (x: (NonTerminal[WithTime[Double]], Boolean)): Double = {
      val (state, bool) = x
      if (bool) 0.0 else f(state)
    }
    
    val f0: ((NonTerminal[WithTime[Double]], Boolean)) => Double =
      (x: (NonTerminal[WithTime[Double]], Boolean)) => {
        val (state, bool) = x
        if (bool) payoff(state.state.time, state.state.state) else 0.0
      }
    
    f0 +: featureFunctions.map(functor)
  }
  
  def transitions(
    alpha: Double = 1.0,
    expectationSamples: Int = 1000
  ): LazyList[LazyList[ActionStep[WithTime[Double], Boolean]]] = {
    val initialDistribution = this.initialStateDistribution
    val policy: Policy[WithTime[Double], Boolean] = (state: NonTerminal[WithTime[Double]]) => {
      val (price, time) = state.state.pair
      val currentPayoff = payoff(time, price)
      val probability = if (currentPayoff <= 0) 0.0 else getLogisticFunction(alpha)(currentPayoff / price)
      Bernoulli(p = probability)
    }
    val mdp = getMDP(expectationSamples)
    mdp.actionTraces(initialDistribution, policy)
  }
  
  def initialStateDistribution: Distribution[NonTerminal[WithTime[Double]]] = {
    val initialNoiseDistribution = Gaussian(
      mu = -0.5 * pow(spotPriceFrac, 2),
      sigma = spotPriceFrac
    )
    initialNoiseDistribution.map(noise => NonTerminal(WithTime(spotPrice * exp(noise))))
  }
  
  def getMDP(expectationSamples: Int = 1000): MarkovDecisionProcess[WithTime[Double], Boolean] = {
    
    def stateRewardSampler(
      state: NonTerminal[WithTime[Double]],
      exercise: Boolean
    ): (State[WithTime[Double]], Double) = {
      val withTime = state.state
      val (price, time) = withTime.pair
      if (exercise || time == numSteps - 1)
        (Terminal(withTime.stepTime(0.0)), payoff(time, price))
      else {
        val distribution = Gaussian(
          mu = log(price) + (rate - 0.5 * vol * vol) * dt,
          sigma = vol * sqrtDt)
        val nextPrice = exp(distribution.sample)
        (NonTerminal(withTime.stepTime(nextPrice)), 0.0)
      }
    }
    
    new MarkovDecisionProcess[WithTime[Double], Boolean] {
      override def step(
        state: NonTerminal[WithTime[Double]],
        action: Boolean
      ): SampledDistribution[(State[WithTime[Double]], Double)] = {
        
        SampledDistribution(
          sampler = () => stateRewardSampler(state, action),
          expectationSamples = expectationSamples
        )
      }
      
      override def actions(state: NonTerminal[WithTime[Double]]): Iterable[Boolean] = Seq(true, false)
    }
  }
  
  protected def getValueFunctionAndPolicyFromLeastSquaresQValueFunction(
    qValueFunction: LinearFunctionApproximation[(NonTerminal[WithTime[Double]], Boolean)]
  ): (LinearFunctionApproximation[NonTerminal[WithTime[Double]]], DeterministicPolicy[WithTime[Double], Boolean]) = {
    
    val deterministicPolicy: DeterministicPolicy[WithTime[Double], Boolean] = { (state: WithTime[Double]) =>
      Seq(true, false).maxBy(b => qValueFunction(NonTerminal(state) -> b))
    }
    
    val qFeatureFunctions: Seq[((NonTerminal[WithTime[Double]], Boolean)) => Double] = qValueFunction.featureFunctions
    
    def functor(f: ((NonTerminal[WithTime[Double]], Boolean)) => Double)(x: NonTerminal[WithTime[Double]]): Double = {
      val optimalAction = deterministicPolicy.actionForState(x.state)
      f(x -> optimalAction)
    }
    
    val vFeatureFunctions: Seq[NonTerminal[WithTime[Double]] => Double] = qFeatureFunctions.map(functor)
    val valueFunction: LinearFunctionApproximation[NonTerminal[WithTime[Double]]] =
      LinearFunctionApproximation.create(
        featureFunctions = vFeatureFunctions,
        weightsOption = Some(qValueFunction.weights),
        regularizationCoefficient = qValueFunction.regularizationCoefficient
      )
    
    (valueFunction, deterministicPolicy)
  }
  
}

object OptimalExerciseRL {
  
  def apply(
    spotPrice: Double,
    payoff: (Double, Double) => Double,
    expiry: Double,
    rate: Double,
    vol: Double,
    numSteps: Int,
    spotPriceFrac: Double
  ): OptimalExerciseRL =
    new OptimalExerciseRL(spotPrice, payoff, expiry, rate, vol, numSteps, spotPriceFrac)
  
  def defaultAdamGradient: AdamGradient = AdamGradient(
    learningRate = 0.1,
    decay1 = 0.9,
    decay2 = 0.999
  )
  
}

object OptimalExerciseRLApp extends App {
  
  val logger: Logger = Logger("OptimalExerciseRLApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val spotPrice: Double = 100.0
  val strike: Double = 105.0
  val expiry: Double = 1.0
  val rate: Double = 0.05
  val vol: Double = 0.25
  val numSteps: Int = 10
  val spotPriceFrac: Double = 0.02
  
  val optimalExerciseRL = OptimalExerciseRL(
    spotPrice = spotPrice,
    payoff = (_: Double, x: Double) => math.max(strike - x, 0),
    expiry = expiry,
    rate = rate,
    vol = vol,
    numSteps = numSteps,
    spotPriceFrac = spotPriceFrac
  )
  
  val laguerrePolynomials: Seq[Double => Double] = Utils.laguerrePolynomials.take(3)
  val dt = expiry / numSteps
  val pi = math.Pi
  
  val featureFunctions: Seq[NonTerminal[WithTime[Double]] => Double] =
    Seq(
      (x: NonTerminal[WithTime[Double]]) =>
        if (x.state.time < numSteps) 1.0 else 0.0,
      (x: NonTerminal[WithTime[Double]]) =>
        if (x.state.time < numSteps) cos(-x.state.time * pi * dt / (2 * expiry)) else 0.0,
      (x: NonTerminal[WithTime[Double]]) =>
        if (x.state.time < numSteps) log(expiry - x.state.time * dt) else 0.0,
      (x: NonTerminal[WithTime[Double]]) =>
        if (x.state.time < numSteps) pow(x.state.time * dt / expiry, 2) else 0.0
    ) ++ laguerrePolynomials.map { f =>
      (x: NonTerminal[WithTime[Double]]) =>
        if (x.state.time < numSteps) log(1 + exp(-0.5 * x.state.state / strike)) * f(x.state.state / strike) else 0.0
    }
  
  //logger.info(f"Sample transitions:\n${optimalExerciseRL.transitions().flatten.take(30).toList.mkString("\n")}")
  
  val numTransitions = 1e5.toInt
  val numIterations = 10
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val (lspiValueFunction, lspiPolicy) = optimalExerciseRL.lspiValueFunctionAndPolicy(
    numTransitions = numTransitions,
    numIterations = numIterations,
    featureFunctions = featureFunctions
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  val prices = DenseVector.rangeD(0.0, 150.0)
  
  (0 to numSteps).reverse.foreach { timeStep =>
    if (timeStep == 0 || timeStep == (numSteps - 1) || timeStep == (numSteps / 2)) {
      val optimalValues = optimalExerciseRL.optimalValueCurve(timeStep, lspiValueFunction, prices.toScalaVector())
      val exerciseValues = optimalExerciseRL.exerciseCurve(timeStep, prices.toScalaVector())
      val fig = Figure(s"Option Value at time step $timeStep")
      val p = fig.subplot(0)
      p += plot(prices, exerciseValues, name = "Payoff Function")
      p += plot(prices, optimalValues, name = "Approximate Optimal Value Function")
      p.legend = true
    }
  }
  
  (0 to numSteps).reverse.foreach { timeStep =>
    val state = NonTerminal(WithTime(spotPrice, timeStep))
    val europeanPrice: Double = optimalExerciseRL.europeanPutPrice(strike, timeStep)
    val lspiPrice: Double = lspiValueFunction(state)
    val lspiAction: Boolean = lspiPolicy.actionForState(state.state)
    logger.info(f"Time Step $timeStep: LSPI Price = $lspiPrice%1.3f, LSPI Action = $lspiAction, European Price = $europeanPrice%1.3f")
  }
  
}
