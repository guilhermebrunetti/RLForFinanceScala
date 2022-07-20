package rl.chapter8

import java.util.Locale

import breeze.linalg.DenseVector
import breeze.numerics._
import breeze.plot._
import com.typesafe.scalalogging.Logger
import rl.ApproximateDynamicProgramming._
import rl.utils._
import rl.{DeterministicPolicy, NonTerminal, Terminal, _}

/**
 * Optimal Exercise with Backward Induction when the underlying
 * price follows a lognormal process
 */
class OptimalExerciseBackwardInduction(
  val spotPrice: Double,
  val payoff: Double => Double,
  val expiry: Double,
  val rate: Double,
  val vol: Double,
  val numSteps: Int,
  val spotPriceFrac: Double
) {
  self =>
  
  val dt: Double = expiry / numSteps
  val sqrtDt: Double = sqrt(dt)
  
  def getValueFunctionAndPolicy(
    featureFunctions: Seq[NonTerminal[Double] => Double],
    regularizationCoefficient: Double,
    numSamples: Int = 1000,
    errorTolerance: Double = 1.0e-8,
  ): Seq[(ValueFunctionApproximation[Double], DeterministicPolicy[Double, Boolean])] = {
    
    val mdpFunctionDistribution: Seq[MDPValueFuncApproxDistribution[Double, Boolean]] =
      (0 until numSteps).map { t =>
        (
          getMDP(t, numSamples),
          getValueFunctionApproximation(t, featureFunctions, regularizationCoefficient),
          getStateDistribution(t)
        )
      }
    
    backwardOptimalValueFunctionAndPolicy(
      mdpFunctionDistribution = mdpFunctionDistribution,
      gamma = exp(-rate * dt),
      numSamples = numSamples,
      errorTolerance = errorTolerance
    )
  }
  
  def getMDP(timeStep: Int, expectationSamples: Int = 1000): MarkovDecisionProcess[Double, Boolean] = {
    
    def stateRewardSampler(
      price: NonTerminal[Double],
      exercise: Boolean
    ): (State[Double], Double) = {
      if (exercise)
        (Terminal(0.0), payoff(price.state))
      else {
        val distribution = Gaussian(
          mu = log(price.state) + (rate - 0.5 * vol * vol) * dt,
          sigma = vol * sqrtDt)
        val nextPrice = exp(distribution.sample)
        (NonTerminal(nextPrice), 0.0)
      }
    }
    
    new MarkovDecisionProcess[Double, Boolean] {
      override def step(state: NonTerminal[Double], action: Boolean): SampledDistribution[(State[Double], Double)] = {
        
        SampledDistribution(
          sampler = () => stateRewardSampler(state, action),
          expectationSamples = expectationSamples
        )
      }
      
      override def actions(state: NonTerminal[Double]): Iterable[Boolean] = Seq(true, false)
    }
  }
  
  def getStateDistribution(timeStep: Int): SampledDistribution[NonTerminal[Double]] = {
    
    val time: Double = timeStep * dt
    
    def stateSampler: NonTerminal[Double] = {
      val initialPriceDistribution = Gaussian(
        mu = -0.5 * pow(spotPriceFrac, 2),
        sigma = spotPriceFrac
      )
      val initialPrice = spotPrice * exp(initialPriceDistribution.sample)
      val logPriceDistribution = Gaussian(
        mu = log(initialPrice) + (rate - 0.5 * vol * vol) * time,
        sigma = vol * sqrt(time)
      )
      val price = exp(logPriceDistribution.sample)
      NonTerminal(price)
    }
    
    SampledDistribution(sampler = () => stateSampler)
  }
  
  def getValueFunctionApproximation(
    timeStep: Int,
    featureFunctions: Seq[NonTerminal[Double] => Double],
    regularizationCoefficient: Double
  ): LinearFunctionApproximation[NonTerminal[Double]] = {
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      regularizationCoefficient = regularizationCoefficient
    )
  }
  
  def optionExerciseBoundary(
    valueFunctions: Seq[ValueFunctionApproximation[Double]],
    strike: Double
  ): Seq[Double] = {
    val prices = DenseVector.rangeD(0.0, strike + 0.1, 0.1).valuesIterator.toSeq
    val initialCurve = valueFunctions.init.map { vf =>
      val cp = optimalValueCurve(vf, prices).toScalaVector
      val ep = exerciseCurve(prices).toScalaVector
      val ll = prices.lazyZip(cp).lazyZip(ep).toSeq.collect {
        case (p, c, e) if e > c => p
      }
      ll.maxOption.getOrElse(0.0)
    }
    val finalBoundary = prices.collect { case p if payoff(p) > 0 => p }.maxOption.getOrElse(0.0)
    initialCurve :+ finalBoundary
  }
  
  def optimalValueCurve(
    functionApproximation: FunctionApproximation[NonTerminal[Double]],
    prices: Iterable[Double]
  ): DenseVector[Double] = {
    functionApproximation.evaluate(prices.map(NonTerminal(_)))
  }
  
  def exerciseCurve(
    prices: Iterable[Double]
  ): DenseVector[Double] = {
    val exerciseValue = prices.map(payoff(_))
    DenseVector(exerciseValue.toArray)
  }
  
  def europeanValueCurve(
    prices: Iterable[Double],
    strike: Double,
    timeStep: Int = 0,
  ): DenseVector[Double] = {
    val putPrices = prices.map(europeanPutPrice(strike, timeStep, _))
    DenseVector(putPrices.toArray)
  }
  
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
  
  
}

object OptimalExerciseBackwardInduction {
  
  def apply(
    spotPrice: Double,
    payoff: Double => Double,
    expiry: Double,
    rate: Double,
    vol: Double,
    numSteps: Int,
    spotPriceFrac: Double
  ): OptimalExerciseBackwardInduction =
    new OptimalExerciseBackwardInduction(
      spotPrice,
      payoff,
      expiry,
      rate,
      vol,
      numSteps,
      spotPriceFrac
    )
}

object OptimalExerciseBackwardInductionApp extends App {
  
  Locale.setDefault(Locale.US) // To print numbers in US format
  val logger: Logger = Logger("OptimalExerciseBackwardInductionApp")
  
  val spotPrice: Double = 100.0
  val strike: Double = 100.0
  val expiry: Double = 1.0
  val rate: Double = 0.05
  val vol: Double = 0.25
  val numSteps: Int = 10
  val spotPriceFrac: Double = 0.02
  
  val optimalExerciseBackwardInduction = OptimalExerciseBackwardInduction(
    spotPrice = spotPrice,
    payoff = (x: Double) => math.max(strike - x, 0),
    expiry = expiry,
    rate = rate,
    vol = vol,
    numSteps = numSteps,
    spotPriceFrac = spotPriceFrac
  )
  
  val regCoefficient: Double = 0.001
  
  val laguerrePolynomials: Seq[Double => Double] = Utils.laguerrePolynomials.take(3)
  
  val featureFunctions: Seq[NonTerminal[Double] => Double] =
    ((_: NonTerminal[Double]) => 1.0) +: laguerrePolynomials.map { f =>
      (x: NonTerminal[Double]) => log(1 + exp(-0.5 * x.state / strike)) * f(x.state / strike)
    }
  
  val valueFunctionsAndPolicies: Seq[(ValueFunctionApproximation[Double], DeterministicPolicy[Double, Boolean])] =
    optimalExerciseBackwardInduction.getValueFunctionAndPolicy(
      featureFunctions = featureFunctions,
      regularizationCoefficient = regCoefficient,
    )
  
  val prices: Seq[Double] = DenseVector.rangeD(0.0, 150.0).valuesIterator.toSeq
  
  valueFunctionsAndPolicies.zipWithIndex.reverse.foreach { case ((vf, policy), timeStep) =>
    val optAction = policy.actionForState(spotPrice)
    val optValue = vf(NonTerminal(spotPrice))
    val europeanValue = optimalExerciseBackwardInduction.europeanPutPrice(strike, timeStep)
    if (timeStep == 0 || timeStep == (numSteps - 1) || timeStep == (numSteps / 2)) {
      val optimalValues = optimalExerciseBackwardInduction.optimalValueCurve(vf, prices)
      val exerciseValues = optimalExerciseBackwardInduction.exerciseCurve(prices)
      val fig = Figure(s"Option Value at time step $timeStep")
      val p = fig.subplot(0)
      p += plot(prices, exerciseValues, name = "Payoff Function")
      p += plot(prices, optimalValues, name = "Approximate Optimal Value Function")
      p.legend = true
    }
    logger.info(f"Time step $timeStep: Optimal Value: $optValue%1.4f, Optimal Action: $optAction, European Value: $europeanValue%1.4f")
  }
  
  logger.info(f"European Put Price at inception: ${optimalExerciseBackwardInduction.europeanPutPrice(strike)}%1.4f")
  
  val (valueFunctions, _) = valueFunctionsAndPolicies.unzip
  val exerciseBoundary = optimalExerciseBackwardInduction.optionExerciseBoundary(valueFunctions, strike)
  val steps = DenseVector.rangeD(0, numSteps)
  
  val fig = Figure("Option Exercise Boundary")
  val p = fig.subplot(0)
  p += plot(steps, exerciseBoundary, name = "Exercise Boundary")
  p.legend = true
  
}
