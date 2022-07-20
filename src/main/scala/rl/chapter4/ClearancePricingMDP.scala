package rl.chapter4

import java.util.Locale

import com.typesafe.scalalogging.Logger
import rl.DynamicProgramming.ValueFunction
import rl.FiniteHorizon._
import rl.FiniteMarkovDecisionProcess.StateActionMapping
import rl.utils.{Categorical, Poisson}
import rl.{FiniteDeterministicPolicy, FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess, FinitePolicy, NonTerminal, WithTime}

class ClearancePricingMDP(
  val initialInventory: Int,
  val timeSteps: Int,
  val priceLambdaPairs: Seq[(Double, Double)]
) {
  
  lazy val finiteHorizonMDP: FiniteMarkovDecisionProcess[WithTime[Int], Int] =
    finiteHorizonMarkovDecisionProcess(singleStepMDP, timeSteps)
  
  lazy val singleStepMDP: FiniteMarkovDecisionProcess[Int, Int] = new FiniteMarkovDecisionProcess[Int, Int] {
    override def actionSortingFunction(x: Int, y: Int): Boolean = x <= y
    override def stateSortingFunction(x: NonTerminal[Int], y: NonTerminal[Int]): Boolean = x.state <= y.state
    
    override def stateActionMap: StateActionMapping[Int, Int] = {
      FiniteMarkovDecisionProcess.processInputMap(inputMap)
    }
  }
  
  lazy val inputMap: Map[Int, Map[Int, Categorical[(Int, Double)]]] = (0 to initialInventory).map { s =>
    s -> pricesDistributionsPairs.zipWithIndex.map { case ((price, distribution), i) =>
      i -> Categorical(
        (0 to s).map { k =>
          (s - k, price * k) -> {
            if (k < s)
              distribution.probabilityMassFunction(k)
            else
              1 - distribution.cumulativeDistributionFunction(s - 1)
          }
        }.toMap
      )
    }.toMap
  }.toMap
  
  val distributions: Seq[Poisson] = priceLambdaPairs.map { case (_, lambda) => Poisson(lambda) }
  val prices: Seq[Double] = priceLambdaPairs.map { case (price, _) => price }
  val pricesDistributionsPairs: Seq[(Double, Poisson)] = priceLambdaPairs.map { case (price, lambda) =>
    (price, Poisson(lambda))
  }
  
  def getValueFunctionForPolicy(
    policy: FinitePolicy[WithTime[Int], Int],
    gamma: Double = 1.0
  ): Seq[ValueFunction[Int]] = {
    val mrp = finiteHorizonMDP.applyFinitePolicy(policy)
    evaluate(unwrapFiniteHorizonMRP(mrp), gamma)
  }
  
  def getOptimalValueFunctionAndPolicy(
    gamma: Double = 1.0
  ): Seq[(ValueFunction[Int], FiniteDeterministicPolicy[Int, Int])] = {
    optimalValueFunctionAndPolicy(unwrapFiniteHorizonMDP(finiteHorizonMDP), gamma)
  }
  
}

object ClearancePricingMDP {
  
  def apply(
    initialInventory: Int,
    timeSteps: Int,
    priceLambdaPairs: Seq[(Double, Double)]
  ): ClearancePricingMDP = new ClearancePricingMDP(initialInventory, timeSteps, priceLambdaPairs)
  
}

object ClearancePricingMDPApp extends App {
  val logger: Logger = Logger("ClearancePricingMDPApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val initialInventory = 12
  val timeSteps = 8
  val priceLambdaPairs = Seq((1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5))
  val gamma = 1.0
  
  val CPMDP = ClearancePricingMDP(
    initialInventory,
    timeSteps,
    priceLambdaPairs
  )
  
  logger.info(s"Clearance Pricing MDP:\n${CPMDP.finiteHorizonMDP}")
  val stationaryPolicy: FiniteDeterministicPolicy[Int, Int] = FiniteDeterministicPolicy(
    (0 to initialInventory).map { s => s -> policyFunction(s) }.toMap
  )
  val singleStepMRP: FiniteMarkovRewardProcess[Int] = CPMDP.singleStepMDP.applyFinitePolicy(stationaryPolicy)
  val valueFunction: Seq[ValueFunction[Int]] = evaluate(
    unwrapFiniteHorizonMRP(finiteHorizonMarkovRewardProcess(singleStepMRP, timeSteps)),
    gamma)
  
  val valueFunctionStr = valueFunction.zipWithIndex.map { case (vf, i) =>
    val vfAsStr = vf
      .toSeq
      .sortBy { case (s, _) => s.state }
      .map { case (s, v) => s"$s -> $v" }
      .mkString("\n")
    s"\tTime step $i:\n---------------\n$vfAsStr"
  }.mkString("\n")
  
  val optimalValueFunctionAndPolicy: Seq[(ValueFunction[Int], FiniteDeterministicPolicy[Int, Int])] =
    CPMDP.getOptimalValueFunctionAndPolicy(gamma)
  
  val optimalValueFunctionAndPolicyStr = optimalValueFunctionAndPolicy.zipWithIndex.map { case ((vf, pi), i) =>
    
    val vfAsStr = vf
      .toSeq
      .sortBy { case (s, _) => s.state }
      .map { case (s, v) => s"$s -> $v" }
      .mkString("\n")
    
    val piAsStr = pi
      .actionMap
      .toSeq
      .sorted
      .map { case (s, a) => s"For State $s: Do action $a" }
      .mkString("\n")
    
    s"\tTime step $i:\n---------------\n$vfAsStr\n$piAsStr"
  }.mkString("\n")
  
  logger.info(s"Value Function for Stationary Policy:\n------------------------------------\n$valueFunctionStr")
  
  logger.info(s"Optimal Value Function and Policy:\n------------------------------------\n$optimalValueFunctionAndPolicyStr")
  
  def policyFunction(x: Int): Int = {
    if (x < 2) 0 else if (x < 5) 1 else if (x < 8) 2 else 3
  }
}
