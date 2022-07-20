package rl.chapter7

import java.util.Locale

import breeze.numerics._
import breeze.linalg._
import rl.ApproximateDynamicProgramming._
import rl.utils.{Choose, Distribution, Gaussian, SampledDistribution}
import rl.{DeterministicPolicy, _}
import com.typesafe.scalalogging.Logger
import java.time.LocalDateTime

class AssetAllocationDiscrete(
  val riskyReturnDistributions: Seq[Distribution[Double]],
  val riskFreeReturns: Seq[Double],
  val utilityFunction: Double => Double,
  val riskyAllocationChoices: Seq[Double],
  val featureFunctions: Seq[(Double, Double) => Double],
  val dnnSpec: DNNSpec,
  val initialWealthDistribution: Distribution[Double]
) {
  self =>
  require(riskyReturnDistributions.size == riskFreeReturns.size,
    s"Return Distributions and riskless returns should have same size"
  )
  
  val timeSteps: Int = riskyReturnDistributions.size
  
  private val riskyDistributionsAndRiskFreeReturns = riskyReturnDistributions.zip(riskFreeReturns)
  
  def qValueFunctions(
    errorTolerance: Double = 1.0e-6,
    numSamples: Int = 300
  ): Seq[QValueFunctionApproximation[Double, Double]] = {
    val initialApproximation = getQValueFunctionApproximation
    
    val mdpQFunctionDistribution: Seq[MDPQValueFuncApproxDistribution[Double, Double]] =
      (0 until timeSteps).map { t =>
        (getMDP(t), initialApproximation, getStateDistribution(t))
      }
    
    backwardOptimalQValueFunction(
      mdpQFunctionDistribution = mdpQFunctionDistribution,
      gamma = 1.0,
      numSamples = numSamples,
      errorTolerance = errorTolerance
    )
  }
  
  /**
   * State is Wealth W_t, Action is investment in risky asset (= x_t)
   * Investment in riskless asset is W_t - x_t
   */
  def getMDP(timeStep: Int): MarkovDecisionProcess[Double, Double] = {
    val distribution: Distribution[Double] = riskyReturnDistributions(timeStep)
    val rate: Double = riskFreeReturns(timeStep)
    
    def stateRewardSampler(
      wealth: NonTerminal[Double],
      allocation: Double
    ): (State[Double], Double) = {
      val riskyReturn = distribution.sample
      val riskFreeAllocation = wealth.state - allocation
      val nextWealth = allocation * (1 + riskyReturn) + riskFreeAllocation * (1 + rate)
      if (timeStep < timeSteps - 1)
        (NonTerminal(nextWealth), 0.0)
      else
        (Terminal(nextWealth), utilityFunction(nextWealth))
    }
    
    new MarkovDecisionProcess[Double, Double] {
      override def step(state: NonTerminal[Double], action: Double): SampledDistribution[(State[Double], Double)] = {
        
        SampledDistribution(
          sampler = () => stateRewardSampler(state, action),
          expectationSamples = 1000
        )
      }
      
      override def actions(state: NonTerminal[Double]): Iterable[Double] = self.riskyAllocationChoices
    }
  }
  
  def getQValueFunctionApproximation: DNNApproximation[(NonTerminal[Double], Double)] = {
    val adamGradient: AdamGradient = AdamGradient(
      learningRate = 0.1,
      decay1 = 0.9,
      decay2 = 0.999
    )
    
    val featureFunctions: Seq[((NonTerminal[Double], Double)) => Double] =
      self.featureFunctions.map(f => { x: (NonTerminal[Double], Double) => f(x._1.state, x._2) })
    
    DNNApproximation.create(
      featureFunctions = featureFunctions,
      dnnSpec = self.dnnSpec,
      adamGradient = adamGradient
    )
  }
  
  def getStateDistribution(timeStep: Int): SampledDistribution[NonTerminal[Double]] = {
    
    val actionDistribution = self.uniformActions
    
    def stateSampler: NonTerminal[Double] = {
      val initialWealth = self.initialWealthDistribution.sample
      val finalWealth = riskyDistributionsAndRiskFreeReturns
        .take(timeStep)
        .foldLeft(initialWealth) { case (wealth, (distribution, rate)) =>
          val allocation = actionDistribution.sample
          val riskyReturn = distribution.sample
          val riskFreeAllocation = wealth - allocation
          allocation * (1 + riskyReturn) + riskFreeAllocation * (1 + rate)
        }
      NonTerminal(finalWealth)
    }
    
    SampledDistribution(sampler = () => stateSampler)
  }
  
  def uniformActions: Choose[Double] = Choose(riskyAllocationChoices)
  
  def valueFunctionsAndPolicies(
    featureFunctions: Seq[NonTerminal[Double] => Double],
    errorTolerance: Double = 1.0e-6,
    numSamples: Int = 300
  ): Seq[(ValueFunctionApproximation[Double], DeterministicPolicy[Double, Double])] = {
    val initialApproximation = getValueFunctionApproximation(featureFunctions)
    
    val mdpFunctionDistribution: Seq[MDPValueFuncApproxDistribution[Double, Double]] =
      (0 until timeSteps).map { t =>
        (getMDP(t), initialApproximation, getStateDistribution(t))
      }
    
    backwardOptimalValueFunctionAndPolicy(
      mdpFunctionDistribution = mdpFunctionDistribution,
      gamma = 1.0,
      numSamples = numSamples,
      errorTolerance = errorTolerance
    )
  }
  
  def getValueFunctionApproximation(
    featureFunctions: Seq[NonTerminal[Double] => Double]
  ): DNNApproximation[NonTerminal[Double]] = {
    val adamGradient: AdamGradient = AdamGradient(
      learningRate = 0.1,
      decay1 = 0.9,
      decay2 = 0.999
    )
    
    DNNApproximation.create(
      featureFunctions = featureFunctions,
      dnnSpec = self.dnnSpec,
      adamGradient = adamGradient
    )
  }
  
}

object AssetAllocationDiscrete {
  def apply(
    riskyReturnDistributions: Seq[Distribution[Double]],
    riskFreeReturns: Seq[Double],
    utilityFunction: Double => Double,
    riskyAllocationChoices: Seq[Double],
    featureFunctions: Seq[(Double, Double) => Double],
    dnnSpec: DNNSpec, initialWealthDistribution: Distribution[Double]
  ): AssetAllocationDiscrete =
    new AssetAllocationDiscrete(
      riskyReturnDistributions,
      riskFreeReturns,
      utilityFunction,
      riskyAllocationChoices,
      featureFunctions,
      dnnSpec,
      initialWealthDistribution
    )
}

object AssetAllocationDiscreteApp extends App {
  
  Locale.setDefault(Locale.US) // To print numbers in US format
  val logger: Logger = Logger("AssetAllocationDiscreteApp")
  
  val steps = 4
  val mu: Double = 0.13
  val sigma: Double = 0.2
  val riskFreeRate: Double = 0.07
  val a: Double = 1.0
  val initialWealth: Double = 1.0
  val initialWealthSigma: Double = 0.1
  
  val excessReturn: Double = mu - riskFreeRate
  val variance: Double = sigma * sigma
  val baseAllocation: Double = excessReturn / (a * variance)
  
  val riskyReturnDistributions: Seq[Gaussian] = (0 until steps).map(_ => Gaussian(mu = mu, sigma = sigma))
  val riskFreeRates: Seq[Double] = Seq.fill(steps)(riskFreeRate)
  val allocationChoices: Seq[Double] = linspace(
    2.0 / 3.0 * baseAllocation,
    4.0 / 3.0 * baseAllocation,
    11
  ).valuesIterator.toSeq
  
  val featureFunctions: Seq[(Double, Double) => Double] = Seq(
    (s, a) => 1.0,
    (s, a) => s,
    (s, a) => a,
    (s, a) => a * a,
  )
  
  val dnnSpec = DNNSpec(
    neurons = Seq.empty[Int],
    bias = false,
    hiddenActivation = x => x,
    hiddenActivationDerivative = x => 1.0,
    outputActivation = x => -signum(a) * exp(-x),
    outputActivationDerivative = x => signum(a) * exp(-x)
  )
  val initialWealthDistribution = Gaussian(mu = initialWealth, sigma = initialWealthSigma)
  def utilityFunction(x: Double): Double = -exp(-a * x) / a
  
  val assetAllocationDiscrete = AssetAllocationDiscrete(
    riskyReturnDistributions = riskyReturnDistributions,
    riskFreeReturns = riskFreeRates,
    utilityFunction = utilityFunction,
    riskyAllocationChoices = allocationChoices,
    featureFunctions = featureFunctions,
    dnnSpec = dnnSpec,
    initialWealthDistribution = initialWealthDistribution
  )
  
  val errorTolerance: Double = 1.0e-5
  val numSamples: Int = 300
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val qValueFunctions = assetAllocationDiscrete.qValueFunctions(errorTolerance, numSamples)
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Backward Induction on Q-Value Function:\n--------------------------------------")
  qValueFunctions.zipWithIndex.foreach { case (qvf, i) =>
    val qValues = allocationChoices
      .map(allocation =>
        allocation -> qvf((NonTerminal(initialWealth), allocation))
      )
    val (optimalAllocation, optimalValue) = qValues.maxBy{case (_, qValue) => qValue}
    val weights = qvf.asInstanceOf[DNNApproximation[(NonTerminal[Double], Double)]].weightMatrices
    logger.info(f"Time $i: Opt Risky Allocation = $optimalAllocation%1.3f, Opt Val = $optimalValue%1.4f")
    logger.info(f"Weights:\n$weights")
  }
  
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
    val weightBias: Double = {
      val x1 = excessReturn * excessReturn * (timeLeft - 1) / (2 * variance)
      val x2 = log(abs(a))
      x1 + x2
    }
    val weightWealth: Double = a * growth * (1 + riskFreeRate)
    val allocationWeight: Double = a * excessReturn * growth
    val allocationSquaredWeight: Double = -variance * pow(a * growth, 2) / 2
    
    logger.info(f"Time $t:")
    logger.info(f"Optimal Risky Allocation = $optAllocation%1.3f, Optimal Value = $optValue%1.4f")
    logger.info(f"Bias Weight = $weightBias%1.3f")
    logger.info(f"Wealth Weight = $weightWealth%1.3f")
    logger.info(f"Allocation Weight = $allocationWeight%1.3f")
    logger.info(f"Allocation Squared Weight = $allocationSquaredWeight%1.3f")
  }
  
}
