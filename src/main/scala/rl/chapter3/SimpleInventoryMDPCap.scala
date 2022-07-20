package rl.chapter3

import java.util.Locale

import com.typesafe.scalalogging.Logger
import rl.DynamicProgramming._
import rl.FiniteMarkovDecisionProcess.{ActionTransition, StateActionMapping}
import rl.FiniteMarkovRewardProcess.StateReward
import rl.chapter2.{InventoryState, InventoryStateOrdering}
import rl.utils.{Categorical, Poisson}
import rl.{FiniteDeterministicPolicy, FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess, NonTerminal, State}

class SimpleInventoryMDPCap(
  val capacity: Int,
  val poissonLambda: Double,
  val holdingCost: Double,
  val stockoutCost: Double
) extends FiniteMarkovDecisionProcess[InventoryState, Int] {
  
  override def stateSortingFunction(x: NonTerminal[InventoryState], y: NonTerminal[InventoryState]): Boolean =
    InventoryStateOrdering.lteq(x.state, y.state)
  
  override def actionSortingFunction(x: Int, y: Int): Boolean = x <= y
  
  override lazy val stateActionMap: StateActionMapping[InventoryState, Int] = {
    (0 to capacity).flatMap { alpha =>
      (0 to (capacity - alpha)).map { beta =>
        val state = InventoryState(alpha, beta)
        val position = state.inventoryPosition
        val baseReward: Double = -holdingCost * state.onHand
        val actionMap: ActionTransition[InventoryState, Int] =
          (0 to (capacity - position)).map { order =>
            val massMap: Map[(State[InventoryState], Double), Double] =
              (0 to position).map { i =>
                val (probability, reward): (Double, Double) = if (i < position)
                  (distribution.probabilityMassFunction(i), baseReward)
                else {
                  val p = 1.0 - distribution.cumulativeDistributionFunction(position - 1)
                  val p2 = distribution.probabilityMassFunction(position)
                  val expectedStockout = p * (poissonLambda - position) + p2 * position
                  val reward = baseReward - stockoutCost * expectedStockout
                  (p, reward)
                }
                (NonTerminal(InventoryState(position - i, order)), reward) -> probability
              }.toMap
            val finiteDistribution: StateReward[InventoryState] = Categorical(massMap)
            order -> finiteDistribution
          }.toMap
        
        NonTerminal(state) -> actionMap
      }
    }.toMap
  }
  
  lazy val distribution: Poisson = Poisson(poissonLambda)
}

object SimpleInventoryMDPCap {
  
  def apply(
    capacity: Int,
    poissonLambda: Double,
    holdingCost: Double,
    stockoutCost: Double
  ): SimpleInventoryMDPCap = new SimpleInventoryMDPCap(capacity, poissonLambda, holdingCost, stockoutCost)
  
}

object SimpleInventoryMDPCapApp extends App {
  
  val logger: Logger = Logger("SimpleInventoryMDPCapApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val capacity = 2
  val poissonLambda = 1.0
  val holdingCost = 1.0
  val stockoutCost = 10.0
  
  val gamma = 0.9
  
  val SIMDPCap = SimpleInventoryMDPCap(capacity, poissonLambda, holdingCost, stockoutCost)
  
  logger.info(s"MDP Transition Map:\n$SIMDPCap")
  
  val finiteDeterministicPolicy: FiniteDeterministicPolicy[InventoryState, Int] = {
    val actionMap: Map[InventoryState, Int] =
      (0 to capacity).flatMap { alpha =>
        (0 to (capacity - alpha)).map { beta =>
          InventoryState(alpha, beta) -> (capacity - (alpha + beta))
        }
      }.toMap
    
    FiniteDeterministicPolicy(actionMap)
  }
  
  
  logger.info(s"Deterministic Policy Map:\n$finiteDeterministicPolicy")
  
  val impliedMRP: FiniteMarkovRewardProcess[InventoryState] = SIMDPCap.applyFinitePolicy(finiteDeterministicPolicy)
  
  logger.info(s"Transition Reward Map:\n$impliedMRP")
  
  val stationaryDistribution = impliedMRP.stationaryDistribution
  
  logger.info(s"Stationary Distribution:\n$stationaryDistribution")
  
  val rewardVector = impliedMRP.rewardVector
  val rewardVectorStr = impliedMRP.rewardVectorToString
  val valueFunction = impliedMRP.valueFunctionVector(gamma)
  val valueFunctionStr = impliedMRP.valueFunctionToString(gamma)
  
  logger.info(s"Reward Vector:\n$rewardVectorStr")
  logger.info(s"Value Function (gamma = $gamma):\n$valueFunctionStr")
  
  val piValueFunction = evaluateMarkovRewardProcessResult(impliedMRP, gamma)
  val piValueFunctionStr = valueFunctionToString(piValueFunction)
  
  logger.info(s"Implied MRP Policy Iteration Value Function (gamma = $gamma):\n$piValueFunctionStr")
  
  val (policyIterationOptimalValueFunction, policyIterationOptimalPolicy) = policyIterationResult(SIMDPCap, gamma)
  val policyIterationOptimalValueFunctionStr = valueFunctionToString(policyIterationOptimalValueFunction)
  logger.info(s"MDP Policy Iteration Optimal Value Function and Optimal Policy (gamma = $gamma):\n$policyIterationOptimalValueFunctionStr\n$policyIterationOptimalPolicy")
  
  val (valueIterationOptimalValueFunction, valueIterationOptimalPolicy) = valueIterationResult(SIMDPCap, gamma)
  val valueIterationOptimalValueFunctionStr = valueFunctionToString(valueIterationOptimalValueFunction)
  logger.info(s"MDP Value Iteration Optimal Value Function and Optimal Policy (gamma = $gamma):\n$valueIterationOptimalValueFunctionStr\n$valueIterationOptimalPolicy")
  
}