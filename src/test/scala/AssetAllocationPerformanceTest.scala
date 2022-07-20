
import java.time.LocalDateTime
import java.util.Locale

import breeze.linalg._
import breeze.numerics.{exp, signum}
import breeze.stats._
import com.typesafe.scalalogging.Logger
import org.scalameter._
import org.scalatest.FunSuite
import rl.ApproximateDynamicProgramming.MDPQValueFuncApproxDistribution
import rl.chapter7.AssetAllocationDiscrete
import rl.utils.Gaussian
import rl.{DNNApproximation, DNNSpec, NonTerminal, State}

import scala.collection.parallel.CollectionConverters._

class AssetAllocationPerformanceTest extends FunSuite {
  
  Locale.setDefault(Locale.US) // To print numbers in US format
  val logger: Logger = Logger("AssetAllocationDiscreteApp")
  
  val steps = 4
  val mu: Double = 0.13
  val sigma: Double = 0.2
  val riskFreeRate: Double = 0.07
  val a: Double = 1.0
  val initWealth: Double = 1.0
  val initWealthStDev: Double = 0.1
  
  val excessReturn: Double = mu - riskFreeRate
  val variance: Double = sigma * sigma
  val baseAllocation: Double = excessReturn / (a * variance)
  
  val riskyReturns: Seq[Gaussian] = (0 until steps).map(_ => Gaussian(mu = mu, sigma = sigma))
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
  
  val dnnSpec: DNNSpec = DNNSpec(
    neurons = Seq.empty[Int],
    bias = false,
    hiddenActivation = x => x,
    hiddenActivationDerivative = x => 1.0,
    outputActivation = x => -signum(a) * exp(-x),
    outputActivationDerivative = x => signum(a) * exp(-x)
  )
  val initialWealthDistribution: Gaussian = Gaussian(mu = initWealth, sigma = initWealthStDev)
  val assetAllocationDiscrete: AssetAllocationDiscrete = AssetAllocationDiscrete(
    riskyReturnDistributions = riskyReturns,
    riskFreeReturns = riskFreeRates,
    utilityFunction = utilityFunction,
    riskyAllocationChoices = allocationChoices,
    featureFunctions = featureFunctions,
    dnnSpec = dnnSpec,
    initialWealthDistribution = initialWealthDistribution
  )
  
  def utilityFunction(x: Double): Double = -exp(-a * x) / a
  
  val initialApproximation: DNNApproximation[(NonTerminal[Double], Double)] =
    assetAllocationDiscrete.getQValueFunctionApproximation
  
  val timeSteps: Int = riskyReturns.size
  
  val mdpQFunctionDistribution: Seq[MDPQValueFuncApproxDistribution[Double, Double]] =
    (0 until timeSteps).map { t =>
      (
        assetAllocationDiscrete.getMDP(t),
        initialApproximation,
        assetAllocationDiscrete.getStateDistribution(t)
      )
    }
  
  val (mdp, approx, dist) = mdpQFunctionDistribution.last
  
  val numSamples: Int = 500
  val errorTolerance: Double = 1.0e-4
  
  val numRepeats: Int = 10
  
  def mrpReturn(stateReward: (State[Double], Double)): Double = {
    val (_, reward) = stateReward
    reward
  }
  
  test("Performance of Generating the Target Sample") {
  
    logger.info(s"Test 1 - Starting computation at ${LocalDateTime.now()}")
  
    val times = (0 until numRepeats).par.map { i =>
      val time: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        dist.samples(numSamples).flatMap { state =>
          mdp.actions(state).map { action =>
            (state, action) -> mdp.step(state, action).expectation(mrpReturn)
          }
        }
      }
      logger.info(f"Test 1 - Iteration $i: Time $time")
      time.value
    }
    logger.info(s"Test 1 - Finished computation at ${LocalDateTime.now()}")
  
    val avgTime = mean(times)
    val stDevTime = stddev(times)
    val minTime = min(times)
    val maxTime = max(times)
    
    logger.info(f"Average time for generating target sample: (Sample size $numRepeats)")
    logger.info(f"Mean: $avgTime%1.3f, StDev: $stDevTime%1.3f, Min: $minTime%1.3f, Max: $maxTime%1.3f")
  }
  
  test("Performance of generating the Target Sample in Alternative way") {
    
    logger.info(s"Test 2 - Starting computation at ${LocalDateTime.now()}")
    
    val times = (0 until numRepeats).par.map { i =>
      val time: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        dist.samples(numSamples).par.flatMap { state =>
          mdp.actions(state).map { action =>
            (state, action) -> mdp.step(state, action).expectationPar(mrpReturn)
          }
        }
      }
      logger.info(f"Test 2 - Iteration $i: Time $time")
      time.value
    }
    logger.info(s"Test 2 - Finished computation at ${LocalDateTime.now()}")
    
    val avgTime = mean(times)
    val stDevTime = stddev(times)
    val minTime = min(times)
    val maxTime = max(times)
    
    logger.info(f"Average time for generating target sample with parallel collections: (Sample size $numRepeats)")
    logger.info(f"Mean: $avgTime%1.3f, StDev: $stDevTime%1.3f, Min: $minTime%1.3f, Max: $maxTime%1.3f")
  }
  
  test("Performance of Training the Neural Network"){
  
    val numSamples: Int = 300
    val errorTolerance: Double = 1.0e-4
  
    val target: Seq[((NonTerminal[Double], Double), Double)] =
      dist.samples(numSamples).flatMap { state =>
        mdp.actions(state).map { action =>
          (state, action) -> mdp.step(state, action).expectation(mrpReturn)
        }
      }
  
    logger.info(s"Test 3 - Starting computation at ${LocalDateTime.now()}")
    
    val times = (0 until numRepeats).par.map{ i =>
      val time: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        approx.solve(target, Some(errorTolerance))
      }
      logger.info(f"Test 3 - Iteration $i: Time $time")
      time.value
    }
  
    logger.info(s"Test 3 - Finished computation at ${LocalDateTime.now()}")
    
    val avgTime = mean(times)
    val stDevTime = stddev(times)
    val minTime = min(times)
    val maxTime = max(times)
  
    logger.info(f"Training the Neural Network: (Sample size $numRepeats)")
    logger.info(f"Mean: $avgTime%1.3f, StDev: $stDevTime%1.3f, Min: $minTime%1.3f, Max: $maxTime%1.3f")
  }
  
  
}
