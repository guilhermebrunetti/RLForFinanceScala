package rl.chapter9

import java.util.Locale

import com.typesafe.scalalogging.Logger
import rl.ApproximateDynamicProgramming._
import rl.utils._
import rl.{DeterministicPolicy, LinearFunctionApproximation, MarkovDecisionProcess, NonTerminal, State}
import java.time.LocalDateTime

case class PriceAndShares(
  price: Double,
  shares: Int
)

/**
 * shares refers to the total number of shares N to be sold over
 * T time steps.
 *
 * numSteps refers to the number of time steps T.
 *
 * avgExecPriceImpact refers to the time-sequenced functions g_t
 * that gives the average reduction in the price obtained by the
 * Market Order at time t due to eating into the Buy LOs. g_t is
 * a function of PriceAndShares that represents the pair of Price P_t
 * and MO size N_t. Sales Proceeds = N_t*(P_t - g_t(P_t, N_t)).
 *
 * priceDynamics refers to the time-sequenced functions f_t that
 * represents the price dynamics: P_{t+1} ~ f_t(P_t, N_t). f_t
 * outputs a distribution of prices.
 *
 * utilityFunction refers to the Utility of Sales proceeds function,
 * incorporating any risk-aversion.
 *
 * discountFactor refers to the discount factor gamma.
 *
 * funcApproximation refers to the FunctionApprox required to approximate
 * the Value Function for each time step.
 *
 * initialPriceDistribution refers to the distribution of prices
 * at time 0 (needed to generate the samples of states at each time step,
 * needed in the approximate backward induction algorithm).
 */
class OptimalOrderExecution(
  val shares: Int,
  val numSteps: Int,
  val avgExecPriceImpact: Seq[PriceAndShares => Double],
  val priceDynamics: Seq[PriceAndShares => Distribution[Double]],
  val utilityFunction: Double => Double,
  val discountFactor: Double = 1.0,
  val functionApproximation: ValueFunctionApproximation[PriceAndShares],
  val initialPriceDistribution: Distribution[Double],
) {
  
  def getValueFunctionAndPolicy(
    numSamples: Int = 1000,
    errorTolerance: Double = 1.0e-6,
  ): Seq[(ValueFunctionApproximation[PriceAndShares], DeterministicPolicy[PriceAndShares, Int])] = {
    
    val mdpFunctionDistribution: Seq[MDPValueFuncApproxDistribution[PriceAndShares, Int]] =
      (0 until numSteps).map { t =>
        (
          getMDP(t, numSamples),
          functionApproximation,
          getStateDistribution(t, numSamples)
        )
      }
    
    backwardOptimalValueFunctionAndPolicy(
      mdpFunctionDistribution = mdpFunctionDistribution,
      gamma = discountFactor,
      numSamples = numSamples,
      errorTolerance = errorTolerance
    )
  }
  
  def getMDP(timeStep: Int, numSamples: Int = 1000): MarkovDecisionProcess[PriceAndShares, Int] = {
    
    def stateRewardSampler(
      priceAndShares: NonTerminal[PriceAndShares],
      sell: Int
    ): (State[PriceAndShares], Double) = {
      
      val currentPrice = priceAndShares.state.price
      
      val sellOrder: PriceAndShares = PriceAndShares(
        price = priceAndShares.state.price,
        shares = sell
      )
      
      val nextPrice: Double = priceDynamics(timeStep)(sellOrder).sample
      val remainingShares = priceAndShares.state.shares - sell
      
      val nextState: PriceAndShares = PriceAndShares(
        price = nextPrice,
        shares = remainingShares
      )
      
      val impact: Double = avgExecPriceImpact(timeStep)(sellOrder)
      val reward: Double = utilityFunction(
        sell * (currentPrice - impact)
      )
      
      (NonTerminal(nextState), reward)
    }
    
    new MarkovDecisionProcess[PriceAndShares, Int] {
      override def step(state: NonTerminal[PriceAndShares], action: Int):
      SampledDistribution[(State[PriceAndShares], Double)] = {
        
        SampledDistribution(
          sampler = () => stateRewardSampler(state, action),
          expectationSamples = numSamples
        )
      }
      
      override def actions(state: NonTerminal[PriceAndShares]): Seq[Int] = {
        val currentShares = state.state.shares
        if (timeStep == numSteps - 1)
          Seq(currentShares)
        else
          0 to currentShares
      }
    }
  }
  
  def getStateDistribution(timeStep: Int, numSamples: Int = 1000):
  SampledDistribution[NonTerminal[PriceAndShares]] = {
    
    def stateSampler: NonTerminal[PriceAndShares] = {
      val initialPrice = initialPriceDistribution.sample
      val totalShares = shares
      val initialState = PriceAndShares(initialPrice, totalShares)
      val finalState = priceDynamics
        .take(timeStep)
        .foldLeft(initialState) { case (priceAndShares, dynamic) =>
          val currentPrice = priceAndShares.price
          val currentShares = priceAndShares.shares
          val sellSize = Choose(0 to currentShares).sample
          val sellOrder = PriceAndShares(currentPrice, sellSize)
          val nextPrice = dynamic(sellOrder).sample
          val remainingShares = currentShares - sellSize
          PriceAndShares(nextPrice, remainingShares)
        }
      NonTerminal(finalState)
    }
    
    SampledDistribution(sampler = () => stateSampler)
  }
}

object OptimalOrderExecution {
  
  def apply(
    shares: Int,
    numSteps: Int,
    avgExecPriceImpact: Seq[PriceAndShares => Double],
    priceDynamics: Seq[PriceAndShares => Distribution[Double]],
    utilityFunction: Double => Double,
    discountFactor: Double = 1.0,
    functionApproximation: ValueFunctionApproximation[PriceAndShares],
    initialPriceDistribution: Distribution[Double]
  ): OptimalOrderExecution =
    new OptimalOrderExecution(
      shares,
      numSteps,
      avgExecPriceImpact,
      priceDynamics,
      utilityFunction,
      discountFactor,
      functionApproximation,
      initialPriceDistribution
    )
}

object OptimalOrderExecutionApp extends App {
  
  Locale.setDefault(Locale.US) // To print numbers in US format
  val logger: Logger = Logger("OptimalOrderExecutionApp")
  
  val shares: Int = 100
  val numSteps: Int = 5
  val initialPriceMean: Double = 100.0
  val initialPriceStDev: Double = 10.0
  
  val alpha: Double = 0.03 // Linear Permanent price impact coefficient
  val beta: Double = 0.05 // Linear Temporary price impact coefficient
  val sigma: Double = 0.10 // Normal Volatility
  val avgExecPriceImpact: Seq[PriceAndShares => Double] = Seq.fill(numSteps)(
    (ps: PriceAndShares) => beta * ps.shares
  )
  
  val priceDynamics: Seq[PriceAndShares => Distribution[Double]] = Seq.fill(numSteps)(
    (ps: PriceAndShares) => Gaussian(mu = ps.price - alpha * ps.shares, sigma = sigma)
  )
  
  val initialPriceDistribution: Distribution[Double] =
    Gaussian(mu = initialPriceMean, sigma = initialPriceStDev)
  
  val featureFunctions: Seq[NonTerminal[PriceAndShares] => Double] = Seq(
    ps => ps.state.price * ps.state.shares,
    ps => ps.state.shares * ps.state.shares
  )
  
  val functionApproximation: LinearFunctionApproximation[NonTerminal[PriceAndShares]] =
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions
    )
  val utilityFunction: Double => Double = x => x
  
  val optimalOrderExecution = OptimalOrderExecution(
    shares = shares,
    numSteps = numSteps,
    avgExecPriceImpact = avgExecPriceImpact,
    priceDynamics = priceDynamics,
    functionApproximation = functionApproximation,
    initialPriceDistribution = initialPriceDistribution,
    utilityFunction = utilityFunction
  )
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val valueFunctionsAndPolicies:
    Seq[(ValueFunctionApproximation[PriceAndShares], DeterministicPolicy[PriceAndShares, Int])] =
    optimalOrderExecution.getValueFunctionAndPolicy(
      numSamples = 1000
    )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  val initialState = PriceAndShares(initialPriceMean, shares)
  logger.info(s"Backward Induction: VF And Policy\n---------------------------------")
  
  valueFunctionsAndPolicies.zipWithIndex.foreach { case ((vf, policy), t) =>
    val lfa = vf.asInstanceOf[LinearFunctionApproximation[NonTerminal[PriceAndShares]]]
    val optimalAction = policy.actionForState(initialState)
    val optimalValue = vf(NonTerminal(initialState))
    logger.info(f"Time $t: Optimal Action: $optimalAction, Optimal Value: $optimalValue%1.4f")
    logger.info(f"Optimal Weights below:\n${lfa.weights.weights}")
  }
  
  logger.info(s"Analytical Solution:\n---------------------------------")
  (0 until numSteps).foreach { t =>
    val left = numSteps - t
    val optimalAction: Double = shares.toDouble / left.toDouble
    val weight1: Double = 1.0
    val weight2: Double = -(2 * beta + alpha * (left - 1)) / (2 * left)
    val optimalValue: Double = weight1 * initialPriceMean * shares + weight2 * shares * shares
    logger.info(f"Time $t: Optimal Action: $optimalAction, Optimal Value: $optimalValue%1.4f")
    logger.info(f"Optimal Weights: Weight1: $weight1, Weight2: $weight2")
  }
  
}
