package rl.chapter8

import java.util.Locale

import breeze.numerics._
import breeze.plot._
import com.typesafe.scalalogging.Logger
import org.jfree.chart.axis.NumberTickUnit
import rl.DynamicProgramming.ValueFunction
import rl.FiniteHorizon._
import rl.FiniteMarkovDecisionProcess.StateActionMapping
import rl.FiniteMarkovRewardProcess.StateReward
import rl.utils._
import rl.{FiniteDeterministicPolicy, NonTerminal, Terminal}

class OptimalExerciseBinaryTree(
  val spotPrice: Double,
  val payoff: (Double, Double) => Double,
  val expiry: Double,
  val rate: Double,
  val vol: Double,
  val numSteps: Int
) {
  require(vol / sqrt(expiry / numSteps) / rate > 1,
    s"No-arbitrage condition violated: (vol / sqrt(expiry / numSteps) / rate) should be > 1, got instead ${vol / sqrt(expiry / numSteps) / rate} "
  )
  
  val dt: Double = expiry / numSteps
  val sqrtDt: Double = sqrt(dt)
  
  def europeanPrice(
    isCall: Boolean,
    strike: Double
  ): Double = BlackScholesUtils.europeanOptionPrice(
    spotPrice = spotPrice,
    strike = strike,
    expiry = expiry,
    vol = vol,
    rate = rate,
    isCall = isCall
  )
  
  def getOptimalValueFunctionAndPolicy: Seq[(ValueFunction[Int], FiniteDeterministicPolicy[Int, Boolean])] = {
    val upFactor = exp(vol * sqrtDt)
    val accrual = exp(rate * dt)
    val upProb = (accrual * upFactor - 1) / (upFactor * upFactor - 1)
    val downProb = 1.0 - upProb
    val steps: Seq[StateActionMapping[Int, Boolean]] = (0 to numSteps).map { i =>
      (0 to i).map { j =>
        NonTerminal(j) -> Map[Boolean, StateReward[Int]](
          true -> Constant((Terminal(-1), payoff(i * dt, statePrice(i, j)))),
          false -> Categorical(Map(
            (NonTerminal(j + 1), 0.0) -> upProb,
            (NonTerminal(j), 0.0) -> downProb,
          ))
        )
      }.toMap
    }
    optimalValueFunctionAndPolicy(steps = steps, gamma = exp(-rate * dt))
  }
  
  def statePrice(i: Int, j: Int): Double = {
    spotPrice * exp((2 * j - i) * vol * sqrtDt)
  }
  
  def optionExerciseBoundary(
    policies: Seq[FiniteDeterministicPolicy[Int, Boolean]],
    isCall: Boolean
  ): Seq[(Double, Double)] = {
    policies.zipWithIndex.map { case (policy, i) =>
      val exercisePoints: Seq[Int] = (0 to i).filter(j =>
        policy.actionForState(j) && payoff(i * dt, statePrice(i, j)) > 0
      )
      val boundaryPointOpt: Option[Int] = exercisePoints.length match {
        case 0 => None
        case _ => Some(if (isCall) exercisePoints.min else exercisePoints.max)
      }
      val boundaryPriceOpt = boundaryPointOpt.map(statePrice(i, _))
      
      (i * dt, boundaryPriceOpt)
    }.collect { case (x, Some(y)) => (x, y) }
  }
  
  def copy(
    spotPrice: Double = this.spotPrice,
    payoff: (Double, Double) => Double = this.payoff,
    expiry: Double = this.expiry,
    rate: Double = this.rate,
    vol: Double = this.vol,
    numSteps: Int = this.numSteps
  ): OptimalExerciseBinaryTree = {
    OptimalExerciseBinaryTree(spotPrice, payoff, expiry, rate, vol, numSteps)
  }
  
}

object OptimalExerciseBinaryTree {
  
  def apply(
    spotPrice: Double,
    payoff: (Double, Double) => Double,
    expiry: Double,
    rate: Double,
    vol: Double,
    numSteps: Int
  ): OptimalExerciseBinaryTree =
    new OptimalExerciseBinaryTree(spotPrice, payoff, expiry, rate, vol, numSteps)
  
}

object OptimalExerciseBinaryTreeApp extends App {
  
  Locale.setDefault(Locale.US) // To print numbers in US format
  val logger: Logger = Logger("OptimalExerciseBinaryTreeApp")
  
  val spotPrice: Double = 100.0
  val strike: Double = 100.0
  val expiry: Double = 1.0
  val rate: Double = 0.05
  val vol: Double = 0.25
  val numSteps: Int = 300
  
  val optExBinTreePut = OptimalExerciseBinaryTree(
    spotPrice = spotPrice,
    payoff = optionPayoff(isCall = false, strike),
    expiry = expiry,
    rate = rate,
    vol = vol,
    numSteps = numSteps
  )
  
  val optExBinTreeCall = optExBinTreePut.copy(payoff = optionPayoff(isCall = true, strike))
  
  val (valueFunctionsPut, policiesPut) = optExBinTreePut.getOptimalValueFunctionAndPolicy.unzip
  val (valueFunctionsCall, policiesCall) = optExBinTreeCall.getOptimalValueFunctionAndPolicy.unzip
  
  def optionPayoff(isCall: Boolean, strike: Double)(time: Double, spotPrice: Double): Double =
    BlackScholesUtils.optionPayoff(spotPrice, strike, isCall)
  
  val europeanPutPrice = optExBinTreePut.europeanPrice(isCall = false, strike)
  val americanPutPrice = valueFunctionsPut.head(NonTerminal(0))
  val europeanCallPrice = optExBinTreeCall.europeanPrice(isCall = true, strike)
  val americanCallPrice = valueFunctionsCall.head(NonTerminal(0))
  
  logger.info(f"European Put Price: $europeanPutPrice")
  logger.info(f"American Put Price: $americanPutPrice")
  logger.info(f"European Call Price: $europeanCallPrice")
  logger.info(f"American Call Price: $americanCallPrice")
  
  val (steps, exerciseBoundary) = optExBinTreePut.optionExerciseBoundary(policiesPut, isCall = false).unzip
  val fig = Figure("Option Exercise Boundary")
  val p = fig.subplot(0)
  p += plot(steps, exerciseBoundary, name = "Exercise Boundary")
  p.legend = true
  p.xaxis.setTickUnit(new NumberTickUnit(0.1))
  
}
