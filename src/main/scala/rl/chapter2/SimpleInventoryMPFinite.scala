package rl.chapter2

import java.util.Locale

import breeze.linalg
import com.typesafe.scalalogging.Logger
import rl.FiniteMarkovProcess.Transition
import rl.utils.{Categorical, Poisson}
import rl.{FiniteMarkovProcess, NonTerminal}

case class InventoryState(onHand: Int, onOrder: Int) {
  def inventoryPosition: Int = onHand + onOrder
  
  def pair: (Int, Int) = (onHand, onOrder)
}

object InventoryStateOrdering extends Ordering[InventoryState] {
  override def compare(x: InventoryState, y: InventoryState): Int =
    math.Ordering[(Int, Int)].compare(x.pair, y.pair)
}

class SimpleInventoryMPFinite(
  val capacity: Int,
  val poissonLambda: Double
) extends FiniteMarkovProcess[InventoryState] {
  
  override def sortingFunction(x: NonTerminal[InventoryState], y: NonTerminal[InventoryState]): Boolean =
    InventoryStateOrdering.lteq(x.state, y.state)
  
  lazy val distribution: Poisson = Poisson(poissonLambda)
  
  override lazy val transitionMap: Transition[InventoryState] = {
    FiniteMarkovProcess.transitionsFromMap(transitionSeq.toMap)
  }
  
  protected def transitionSeq: Seq[(InventoryState, Categorical[InventoryState])] =
    (0 to capacity).flatMap { alpha =>
      (0 to (capacity - alpha)).map { beta =>
        val state = InventoryState(alpha, beta)
        val position = state.inventoryPosition
        val beta1 = capacity - position
        val massMap: Map[InventoryState, Double] = (0 to position).map { i =>
          val probability = if (i < position)
            distribution.probabilityMassFunction(i)
          else
            1.0 - distribution.cumulativeDistributionFunction(position - 1)
          InventoryState(position - i, beta1) -> probability
        }.toMap
        
        val finiteDistribution = Categorical(massMap)
        state -> finiteDistribution
      }
    }
  
}

object SimpleInventoryMPFinite {
  
  def apply(capacity: Int, poissonLambda: Double): SimpleInventoryMPFinite = {
    new SimpleInventoryMPFinite(capacity, poissonLambda)
  }
}

object SimpleInventoryMPApp extends App {
  
  val logger: Logger = Logger("SimpleInventoryMPApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val capacity = 2
  val poissonLambda = 1.0
  
  val SIMP = SimpleInventoryMPFinite(capacity, poissonLambda)
  val nonTerminalStates = SIMP.nonTerminalStates
  
  logger.info(s"Non Terminal States:\n${nonTerminalStates.mkString("\n")}")
  
  logger.info(s"Transitions:\n$SIMP")
  logger.info(s"Transition Matrix:\n${linalg.convert(SIMP.transitionMatrix, Float)}")
  
  val stationaryDistribution = SIMP.stationaryDistribution
  
  logger.info(s"Stationary Distribution:\n$stationaryDistribution")
  
  val initialDistribution = stationaryDistribution.map(NonTerminal(_))
  val trajectory = SIMP.simulate(initialDistribution)
  
  val numSteps = 100000
  val empiricalFrequencies = trajectory
    .take(numSteps)
    .toList
    .groupMapReduce(x => x.state)(_ => (1.0 / numSteps).toFloat)(_ + _)
  
  logger.info(s"Empirical Distribution:\n\t${empiricalFrequencies.mkString("\n\t")}")
  
  val firstSteps = 20
  logger.info(s"Sample trajectory of size $firstSteps:\n${trajectory.take(firstSteps).toList.mkString("\n")}")
  
}