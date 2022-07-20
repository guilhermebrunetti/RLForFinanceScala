package rl.chapter2

import java.util.Locale

import breeze.linalg._
import com.typesafe.scalalogging.Logger
import rl.{FiniteMarkovRewardProcess, MarkovRewardProcess, NonTerminal, State}
import rl.FiniteMarkovRewardProcess.RewardTransition
import rl.utils.{Categorical, Distribution, Poisson, SampledDistribution}

class SimpleInventoryMRP(
  val capacity: Int,
  val poissonLambda: Double,
  val holdingCost: Double,
  val stockoutCost: Double
) extends MarkovRewardProcess[InventoryState] {
  
  lazy val distribution: Poisson = Poisson(poissonLambda)
  
  override def transitionReward(
    state: NonTerminal[InventoryState]
  ): Distribution[(State[InventoryState], Double)] = {
    
    def sampleNextState(currentState: State[InventoryState]): (State[InventoryState], Double) = {
      val inventoryState = currentState.state
      val demand = distribution.sample
      val position = inventoryState.inventoryPosition
      val nextState = InventoryState(
        onHand = max(position - demand, 0),
        onOrder = max(capacity - position, 0)
      )
      val reward = -holdingCost * inventoryState.onHand - stockoutCost * max(demand - position, 0)
      (NonTerminal(nextState), reward)
    }
    
    SampledDistribution(() => sampleNextState(state))
  }
}

object SimpleInventoryMRP {
  
  def apply(
    capacity: Int,
    poissonLambda: Double,
    holdingCost: Double,
    stockoutCost: Double
  ): SimpleInventoryMRP =
    new SimpleInventoryMRP(capacity, poissonLambda, holdingCost, stockoutCost)
}

class SimpleInventoryMRPFinite(
  val capacity: Int,
  val poissonLambda: Double,
  val holdingCost: Double,
  val stockoutCost: Double
) extends FiniteMarkovRewardProcess[InventoryState] {
  
  override lazy val transitionRewardMap: RewardTransition[InventoryState] = {
    FiniteMarkovRewardProcess.processInputMap(transitionSeq.toMap)
  }
  lazy val distribution: Poisson = Poisson(poissonLambda)
  
  override def sortingFunction(x: NonTerminal[InventoryState], y: NonTerminal[InventoryState]): Boolean =
    InventoryStateOrdering.lteq(x.state, y.state)
  
  protected def transitionSeq: Seq[(InventoryState, Categorical[(InventoryState, Double)])] =
    (0 to capacity).flatMap { alpha =>
      (0 to (capacity - alpha)).map { beta =>
        val state = InventoryState(alpha, beta)
        val position = state.inventoryPosition
        val beta1 = capacity - position
        val baseReward = -holdingCost * state.onHand
        val massMap: Map[(InventoryState, Double), Double] = (0 to position).map { i =>
          val (probability, reward) = if (i < position)
            (distribution.probabilityMassFunction(i), baseReward)
          else {
            val p = 1.0 - distribution.cumulativeDistributionFunction(position - 1)
            val p2 = distribution.probabilityMassFunction(position)
            val expectedStockout = p * (poissonLambda - position) + p2 * position
            val reward = baseReward - stockoutCost * expectedStockout
            (p, reward)
          }
          (InventoryState(position - i, beta1), reward) -> probability
        }.toMap
        
        val finiteDistribution = Categorical(massMap)
        state -> finiteDistribution
      }
    }
}

object SimpleInventoryMRPFinite {
  
  def apply(
    capacity: Int,
    poissonLambda: Double,
    holdingCost: Double,
    stockoutCost: Double
  ): SimpleInventoryMRPFinite = new SimpleInventoryMRPFinite(capacity, poissonLambda, holdingCost, stockoutCost)
}

object SimpleInventoryMRPApp extends App {
  
  val logger: Logger = Logger("SimpleInventoryMRPApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val capacity = 2
  val poissonLambda = 1.0
  val holdingCost = 1.0
  val stockoutCost = 10.0
  
  val gamma = 0.9
  
  val SIMRP = SimpleInventoryMRP(capacity, poissonLambda, holdingCost, stockoutCost)
  val SIMRPFinite = SimpleInventoryMRPFinite(capacity, poissonLambda, holdingCost, stockoutCost)
  
  logger.info(s"Transition Reward Map:\n$SIMRPFinite")
  
  val stationaryDistribution = SIMRPFinite.stationaryDistribution
  
  logger.info(s"Stationary Distribution:\n$stationaryDistribution")
  
  val rewardVector = SIMRPFinite.rewardVector
  val rewardVectorStr = SIMRPFinite.rewardVectorToString
  val valueFunction = SIMRPFinite.valueFunctionVector(gamma)
  val valueFunctionStr = SIMRPFinite.valueFunctionToString(gamma)
  
  logger.info(s"Reward Vector:\n$rewardVectorStr")
  logger.info(s"Value Function (gamma = $gamma):\n$valueFunctionStr")
  
  val initialDistribution = stationaryDistribution.map(NonTerminal(_))
  val trajectory = SIMRPFinite.simulateReward(initialDistribution)
  
  val numSteps = 100000
  val empiricalRewards = trajectory
    .take(numSteps)
    .toList
    .groupMapReduce(x => x.state)(x => DenseVector[Double](x.reward, 1.0))(_ + _)
    .view
    .mapValues(x => x(0) / x(1))
    .toIndexedSeq
    .sortWith { case ((x, _), (y, _)) => SIMRPFinite.sortingFunction(x, y) }
  
  
  logger.info(s"Empirical rewards (Finite MRP):\n\t${empiricalRewards.mkString("\n\t")}")
  
  val trajectory2 = SIMRP.simulateReward(initialDistribution)
  val empiricalRewards2 = trajectory2
    .take(numSteps)
    .toList
    .groupMapReduce(x => x.state)(x => DenseVector[Double](x.reward, 1.0))(_ + _)
    .view
    .mapValues(x => x(0) / x(1))
    .toIndexedSeq
    .sortWith { case ((x, _), (y, _)) => SIMRPFinite.sortingFunction(x, y) }
  
  logger.info(s"Empirical rewards (MRP):\n\t${empiricalRewards2.mkString("\n\t")}")
  
  val firstSteps = 10
  logger.info(s"Sample trajectory (Finite MRP):\n${trajectory.take(firstSteps).toList.mkString("\n")}")
  logger.info(s"Sample trajectory (MRP):\n${trajectory2.take(firstSteps).toList.mkString("\n")}")
  
  
}
