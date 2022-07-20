package rl.chapter13

import java.time.LocalDateTime
import java.util.Locale

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import breeze.plot._
import com.typesafe.scalalogging.Logger
import rl.ApproximateDynamicProgramming._
import rl.PolicyGradient._
import rl.utils.{Distribution, Gaussian, SampledDistribution}
import AssetAllocationPG._
import rl._

class AssetAllocationPG(
  val riskyReturnDistributions: Seq[Distribution[Double]],
  val riskFreeReturns: Seq[Double],
  val utilityFunction: Double => Double,
  val policyFeatureFunctions: Seq[WithTime[Double] => Double],
  val policyDnnSpec: DNNSpec,
  val policySigma: Double,
  val initialWealthDistribution: Distribution[Double]
) {
  
  val timeSteps: Int = riskyReturnDistributions.size
  
  def reinforce: Iterator[FunctionApproximation[NonTerminal[WithTime[Double]]]] = {
    PolicyGradient.reinforceGaussian(
      markovDecisionProcess = this.getMDP(),
      policyMeanApproximation = this.policyMeanApproximation(),
      initialStateDistribution = this.initialStateDistribution,
      policySigma = policySigma,
      gamma = 1.0,
      episodeLengthTolerance = 1.0e-5
    )
  }
  
  def actorCritic(
    featureFunctions: Seq[((WithTime[Double], Double)) => Double],
    qValueDnnSpec: DNNSpec
  ): Iterator[ActorCriticApproximation[WithTime[Double]]] = {
    
    val qValueFunctionApproximation = this.qValueFunctionApproximation(featureFunctions, qValueDnnSpec)
    
    PolicyGradient.actorCriticGaussian(
      markovDecisionProcess = this.getMDP(),
      policyMeanApproximation = this.policyMeanApproximation(),
      qValueFunctionApproximation = qValueFunctionApproximation,
      initialStateDistribution = this.initialStateDistribution,
      policySigma = policySigma,
      gamma = 1.0,
      maxEpisodeLength = timeSteps
    )
  }
  
  def actorCriticAdvantage(
    qFeatureFunctions: Seq[((WithTime[Double], Double)) => Double],
    qValueDnnSpec: DNNSpec,
    vFeatureFunctions: Seq[WithTime[Double] => Double],
    vFunctionDnnSpec: DNNSpec
  ): Iterator[ActorCriticAdvantageApproximation[WithTime[Double]]] = {
    
    val qValueFunctionApproximation = this.qValueFunctionApproximation(qFeatureFunctions, qValueDnnSpec)
    val valueFunctionApproximation = this.valueFunctionApproximation(vFeatureFunctions, vFunctionDnnSpec)
    
    PolicyGradient.actorCriticAdvantageGaussian(
      markovDecisionProcess = this.getMDP(),
      policyMeanApproximation = this.policyMeanApproximation(),
      qValueFunctionApproximation = qValueFunctionApproximation,
      valueFunctionApproximation = valueFunctionApproximation,
      initialStateDistribution = this.initialStateDistribution,
      policySigma = policySigma,
      gamma = 1.0,
      maxEpisodeLength = timeSteps
    )
  }
  
  def initialStateDistribution: Distribution[NonTerminal[WithTime[Double]]] = {
    initialWealthDistribution.map { wealth => NonTerminal(WithTime(wealth)) }
  }
  
  def getMDP(expectationSamples: Int = 10000): MarkovDecisionProcess[WithTime[Double], Double] = {
    
    def stateRewardSampler(
      state: NonTerminal[WithTime[Double]],
      allocation: Double
    ): (State[WithTime[Double]], Double) = {
      val (wealth, time) = state.state.pair
      val rate = riskFreeReturns(time)
      val riskyReturn = riskyReturnDistributions(time).sample
      val riskFreeAllocation = wealth - allocation
      val nextWealth: Double = allocation * (1 + riskyReturn) + riskFreeAllocation * (1 + rate)
      if (time < timeSteps - 1)
        (NonTerminal(WithTime(nextWealth, time + 1)), 0.0)
      else
        (Terminal(WithTime(nextWealth, time + 1)), utilityFunction(nextWealth))
    }
    
    new MarkovDecisionProcess[WithTime[Double], Double] {
      override def step(
        state: NonTerminal[WithTime[Double]],
        action: Double
      ): SampledDistribution[(State[WithTime[Double]], Double)] = {
        
        SampledDistribution(
          sampler = () => stateRewardSampler(state, action),
          expectationSamples = expectationSamples
        )
      }
      
      override def actions(state: NonTerminal[WithTime[Double]]): Iterable[Double] = Seq.empty[Double]
    }
    
  }
  
  def policyMeanApproximation(
    adamGradient: AdamGradient = AssetAllocationPG.policyAdamGradient
  ): DNNApproximation[NonTerminal[WithTime[Double]]] = {
    
    val transformedFeatures: Seq[NonTerminal[WithTime[Double]] => Double] =
      policyFeatureFunctions.map(f => functor[WithTime[Double]](f))
    
    DNNApproximation.create(
      featureFunctions = transformedFeatures,
      dnnSpec = policyDnnSpec,
      adamGradient = adamGradient
    )
  }
  
  def qValueFunctionApproximation(
    featureFunctions: Seq[((WithTime[Double], Double)) => Double],
    qValueDnnSpec: DNNSpec,
    adamGradient: AdamGradient = AssetAllocationPG.valueFunctionAdamGradient
  ): QValueFunctionApproximation[WithTime[Double], Double] = {
    
    val transformedFeatures: Seq[((NonTerminal[WithTime[Double]], Double)) => Double] =
      featureFunctions.map(f => qFunctor[WithTime[Double], Double](f))
    
    DNNApproximation.create(
      featureFunctions = transformedFeatures,
      dnnSpec = qValueDnnSpec,
      adamGradient = adamGradient
    )
  }
  
  def valueFunctionApproximation(
    featureFunctions: Seq[WithTime[Double] => Double],
    vFunctionDnnSpec: DNNSpec,
    adamGradient: AdamGradient = valueFunctionAdamGradient
  ): ValueFunctionApproximation[WithTime[Double]] = {
    
    val transformedFeatures: Seq[NonTerminal[WithTime[Double]] => Double] =
      featureFunctions.map(f => functor[WithTime[Double]](f))
    
    DNNApproximation.create(
      featureFunctions = transformedFeatures,
      dnnSpec = vFunctionDnnSpec,
      adamGradient = adamGradient
    )
    
  }
  
  def actorCriticTDError(
    vFeatureFunctions: Seq[WithTime[Double] => Double],
    vFunctionDnnSpec: DNNSpec
  ): Iterator[ActorCriticTDErrorApproximation[WithTime[Double]]] = {
    
    val valueFunctionApproximation = this.valueFunctionApproximation(vFeatureFunctions, vFunctionDnnSpec)
    
    PolicyGradient.actorCriticTDErrorGaussian(
      markovDecisionProcess = this.getMDP(),
      policyMeanApproximation = this.policyMeanApproximation(),
      valueFunctionApproximation = valueFunctionApproximation,
      initialStateDistribution = this.initialStateDistribution,
      policySigma = policySigma,
      gamma = 1.0,
      maxEpisodeLength = timeSteps
    )
    
  }
  
}

object AssetAllocationPG {
  
  val policyAdamGradient: AdamGradient = AdamGradient(
    learningRate = 0.003,
    decay1 = 0.9,
    decay2 = 0.999
  )
  
  val valueFunctionAdamGradient: AdamGradient = AdamGradient(
    learningRate = 0.003,
    decay1 = 0.9,
    decay2 = 0.999
  )
  
  def apply(
    riskyReturnDistributions: Seq[Distribution[Double]],
    riskFreeReturns: Seq[Double],
    utilityFunction: Double => Double,
    featureFunctions: Seq[WithTime[Double] => Double],
    dnnSpec: DNNSpec, policySigma: Double,
    initialWealthDistribution: Distribution[Double]
  ): AssetAllocationPG =
    new AssetAllocationPG(
      riskyReturnDistributions,
      riskFreeReturns,
      utilityFunction,
      featureFunctions,
      dnnSpec,
      policySigma,
      initialWealthDistribution
    )
  
  protected def functor[X](f: X => Double)(x: NonTerminal[X]): Double = f(x.state)
  
  protected def qFunctor[X, Y](f: ((X, Y)) => Double)(x: (NonTerminal[X], Y)): Double = f((x._1.state, x._2))
  
}

object AssetAllocationPGApp extends App {
  
  val logger: Logger = Logger("AssetAllocationPGApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val steps: Int = 5
  val mu: Double = 0.13
  val sigma: Double = 0.2
  val riskFreeRate: Double = 0.07
  val a: Double = 1.0
  val initialWealth: Double = 1.0
  val initialWealthSigma: Double = 0.1
  val policySigma: Double = 0.5
  
  val excessReturn: Double = mu - riskFreeRate
  val variance: Double = sigma * sigma
  val baseAllocation: Double = excessReturn / (a * variance)
  
  logger.info(s"Analytical Solution:\n--------------------------------------")
  (0 until steps).foreach { t =>
    val timeLeft: Int = steps - t
    val growth: Double = pow(1 + riskFreeRate, timeLeft - 1)
    val optAllocation: Double = baseAllocation / growth
    
    val optValue: Double = {
      val x1 = -excessReturn * excessReturn * timeLeft / (2 * variance)
      val x2 = -a * growth * (1 + riskFreeRate) * initialWealth
      -exp(x1 + x2) / a
    }
    logger.info(f"Time $t: Optimal Risky Allocation = $optAllocation%1.3f, Optimal Value = $optValue%1.4f")
  }
  
  val riskyReturnDistributions: Seq[Gaussian] = (0 until steps).map(_ => Gaussian(mu = mu, sigma = sigma))
  val riskFreeRates: Seq[Double] = Seq.fill(steps)(riskFreeRate)
  val initialWealthDistribution = Gaussian(mu = initialWealth, sigma = initialWealthSigma)
  
  val utilityFunction: Double => Double = x => -exp(-a * x) / a
  val policyFeatureFunctions: Seq[WithTime[Double] => Double] = Seq(
    (x: WithTime[Double]) => pow(1 + riskFreeRate, x.time)
  )
  
  val policyDnnSpec = DNNSpec(
    neurons = Seq.empty[Int],
    bias = false,
    hiddenActivation = x => x,
    hiddenActivationDerivative = x => 1.0,
    outputActivation = x => x,
    outputActivationDerivative = x => 1.0
  )
  
  val qValueFeatureFunctions: Seq[((WithTime[Double], Double)) => Double] = Seq(
    (x: (WithTime[Double], Double)) => 1.0,
    (x: (WithTime[Double], Double)) => x._1.time.toDouble,
    (x: (WithTime[Double], Double)) => x._1.state * pow(1 + riskFreeRate, -x._1.time),
    (x: (WithTime[Double], Double)) => x._2 * pow(1 + riskFreeRate, -x._1.time),
    (x: (WithTime[Double], Double)) => pow(x._2 * pow(1 + riskFreeRate, -x._1.time), 2),
  )
  
  val qValueDnnSpec = DNNSpec(
    neurons = Seq.empty[Int],
    bias = false,
    hiddenActivation = x => x,
    hiddenActivationDerivative = x => 1.0,
    outputActivation = x => -signum(a) * exp(-x),
    outputActivationDerivative = x => signum(a) * exp(-x)
  )
  
  val valueFeatureFunctions: Seq[WithTime[Double] => Double] = Seq(
    (x: WithTime[Double]) => 1.0,
    (x: WithTime[Double]) => x.time.toDouble,
    (x: WithTime[Double]) => x.state * pow(1 + riskFreeRate, -x.time),
  )
  
  val valueFunctionDnnSpec = DNNSpec(
    neurons = Seq.empty[Int],
    bias = false,
    hiddenActivation = x => x,
    hiddenActivationDerivative = x => 1.0,
    outputActivation = x => -signum(a) * exp(-x),
    outputActivationDerivative = x => signum(a) * exp(-x)
  )
  
  val assetAllocationPG = AssetAllocationPG(
    riskyReturnDistributions = riskyReturnDistributions,
    riskFreeReturns = riskFreeRates,
    utilityFunction = utilityFunction,
    featureFunctions = policyFeatureFunctions,
    dnnSpec = policyDnnSpec,
    policySigma = policySigma,
    initialWealthDistribution = initialWealthDistribution
  )
  
  val numEpisodes: Int = 1e5.toInt
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val reinforcePolicies: Seq[FunctionApproximation[NonTerminal[WithTime[Double]]]] =
    assetAllocationPG.reinforce.take(numEpisodes).toIndexedSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val actorCriticPolicies: Seq[FunctionApproximation[NonTerminal[WithTime[Double]]]] =
    assetAllocationPG.actorCritic(qValueFeatureFunctions, qValueDnnSpec).map(_._1).take(numEpisodes).toIndexedSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val actorCriticAdvantagePolicies: Seq[FunctionApproximation[NonTerminal[WithTime[Double]]]] =
    assetAllocationPG.actorCriticAdvantage(
      qValueFeatureFunctions, qValueDnnSpec, valueFeatureFunctions, valueFunctionDnnSpec
    ).map(_._1).take(numEpisodes).toIndexedSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val actorCriticTDErrorPolicies: Seq[FunctionApproximation[NonTerminal[WithTime[Double]]]] =
    assetAllocationPG.actorCriticTDError(
      valueFeatureFunctions, valueFunctionDnnSpec
    ).map(_._1).take(numEpisodes).toIndexedSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  val averageNumber = 10000
  logger.info(s"REINFORCE Solution:\n--------------------------------------")
  val predictors1 = reinforcePolicies.reverse.take(averageNumber)
  (0 until steps).foreach { t =>
    val state = NonTerminal(WithTime(initialWealth, t))
    val x = predictors1.map(_.apply(state))
    val allocation = mean(x)
    val allocationStDev = stddev(x)
    
    logger.info(f"Time $t: REINFORCE Allocation = $allocation%1.3f (StDev $allocationStDev%1.3f)")
  }
  
  logger.info(s"Actor-Critic Solution:\n--------------------------------------")
  val predictors2 = actorCriticPolicies.reverse.take(averageNumber)
  (0 until steps).foreach { t =>
    val state = NonTerminal(WithTime(initialWealth, t))
    val x = predictors2.map(_.apply(state))
    val allocation = mean(x)
    val allocationStDev = stddev(x)
    
    logger.info(f"Time $t: Actor-Critic Allocation = $allocation%1.3f (StDev $allocationStDev%1.3f)")
  }
  
  logger.info(s"Actor-Critic-Advantage Solution:\n--------------------------------------")
  val predictors3 = actorCriticAdvantagePolicies.reverse.take(averageNumber)
  (0 until steps).foreach { t =>
    val state = NonTerminal(WithTime(initialWealth, t))
    val x = predictors3.map(_.apply(state))
    val allocation = mean(x)
    val allocationStDev = stddev(x)
    
    logger.info(f"Time $t: Actor-Critic-Advantage Allocation = $allocation%1.3f (StDev $allocationStDev%1.3f)")
  }
  
  logger.info(s"Actor-Critic-TD Error Solution:\n--------------------------------------")
  val predictors4 = actorCriticTDErrorPolicies.reverse.take(averageNumber)
  (0 until steps).foreach { t =>
    val state = NonTerminal(WithTime(initialWealth, t))
    val x = predictors4.map(_.apply(state))
    val allocation = mean(x)
    val allocationStDev = stddev(x)
    
    logger.info(f"Time $t: Actor-Critic-TD Error Allocation = $allocation%1.3f (StDev $allocationStDev%1.3f)")
  }
  
  val x = DenseVector.rangeD(0, numEpisodes)
  (0 until math.min(1, steps)).foreach { t =>
    val timeLeft: Int = steps - t
    val growth: Double = pow(1 + riskFreeRate, timeLeft - 1)
    val optAllocation: Double = baseAllocation / growth
    
    val state = NonTerminal(WithTime(initialWealth, timeLeft))
    
    val y0 = DenseVector.fill(numEpisodes)(optAllocation)
    val y1 = DenseVector(reinforcePolicies.map(_.apply(state)).toArray)
    val y2 = DenseVector(actorCriticPolicies.map(_.apply(state)).toArray)
    val y3 = DenseVector(actorCriticAdvantagePolicies.map(_.apply(state)).toArray)
    val y4 = DenseVector(actorCriticTDErrorPolicies.map(_.apply(state)).toArray)
    
    val fig = Figure(f"Action for Initial Wealth at Time $t")
    val p = fig.subplot(0)
    p += plot(x, y0, name = "Optimal Allocation")
    p += plot(x, y1, name = "REINFORCE Allocation")
    p += plot(x, y2, name = "Actor-Critic Allocation")
    p += plot(x, y3, name = "Actor-Critic-Advantage Allocation")
    p += plot(x, y4, name = "Actor-Critic-TD Error Allocation")
    p.legend = true
  }
}
