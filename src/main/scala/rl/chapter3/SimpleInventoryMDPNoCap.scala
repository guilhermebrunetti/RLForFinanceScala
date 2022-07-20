package rl.chapter3

import java.util.Locale

import breeze.linalg._
import com.typesafe.scalalogging.Logger
import rl.chapter2.InventoryState
import rl.utils.{Choose, Distribution, Poisson, SampledDistribution}
import rl.{DeterministicPolicy, MarkovDecisionProcess, NonTerminal, Policy, State, TransitionStep}

case class SimpleInventoryMDPNoCap(
  poissonLambda: Double,
  holdingCost: Double,
  stockoutCost: Double
) extends MarkovDecisionProcess[InventoryState, Int] {
  
  lazy val distribution: Poisson = Poisson(poissonLambda)
  
  override def step(
    state: NonTerminal[InventoryState],
    action: Int
  ): SampledDistribution[(State[InventoryState], Double)] = {
    
    def sampleNextStateReward(
      currentState: NonTerminal[InventoryState],
      order: Int
    ): (State[InventoryState], Double) = {
      val inventoryState = currentState.state
      val demandSample = distribution.sample
      val position = inventoryState.inventoryPosition
      val nextState = InventoryState(
        onHand = max(position - demandSample, 0),
        onOrder = order
      )
      val reward = -holdingCost * inventoryState.onHand - stockoutCost * max(demandSample - position, 0)
      (NonTerminal(nextState), reward)
    }
    
    SampledDistribution(() => sampleNextStateReward(state, action))
  }
  
  override def actions(state: NonTerminal[InventoryState]): Iterable[Int] = LazyList.from(0)
  
  def fractionOfDaysOOS(
    policy: Policy[InventoryState, Int],
    timeSteps: Int,
    numTraces: Int
  ): Double = {
    val impliedMRP = this.applyPolicy(policy)
    val highQuantile = distribution.quantile(0.98)
    
    val initialStateDistribution = Choose(0 to (highQuantile + 1))
      .map(i => NonTerminal(InventoryState(i, 0)))
    val traces: Seq[TransitionStep[InventoryState]] = (0 until numTraces).flatMap { _ =>
      impliedMRP.simulateReward(initialStateDistribution)
        .take(timeSteps)
        .toList
    }
    
    val count: Int = traces.count(step => step.reward < -holdingCost * step.state.state.onHand)
    count.toDouble / (timeSteps * numTraces)
  }
}

case class SimpleInventoryDeterministicPolicy(reorderPoint: Int) extends
  DeterministicPolicy[InventoryState, Int] {
  
  override def actionForState(state: InventoryState): Int = max(reorderPoint - state.inventoryPosition, 0)
}

case class SimpleInventoryStochasticPolicy(reorderPointPoissonMean: Double)
  extends Policy[InventoryState, Int] {
  
  lazy val distribution: Poisson = Poisson(reorderPointPoissonMean)
  
  override def act(state: NonTerminal[InventoryState]): Distribution[Int] = {
    def sampleAction(currentState: NonTerminal[InventoryState]): Int = {
      val reorderPointSample = distribution.sample
      max(reorderPointSample - currentState.state.inventoryPosition, 0)
    }
    
    SampledDistribution(() => sampleAction(state))
  }
}

object SimpleInventoryMDPNoCapApp extends App {
  val logger: Logger = Logger("SimpleInventoryMDPCapApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val poissonLambda = 2.0
  val holdingCost = 1.0
  val stockoutCost = 10.0
  
  val reorderPoint = 8
  val reorderPointPoissonMean = 8.0
  
  val timeSteps = 1000
  val numTraces = 1000
  
  val SIMDPNoCap = SimpleInventoryMDPNoCap(poissonLambda, holdingCost, stockoutCost)
  
  val SIDeterministicPolicy = SimpleInventoryDeterministicPolicy(reorderPoint)
  val SIStochasticPolicy = SimpleInventoryStochasticPolicy(reorderPointPoissonMean)
  
  val ossFractionDP = SIMDPNoCap.fractionOfDaysOOS(SIDeterministicPolicy, timeSteps, numTraces)
  val ossFractionSP = SIMDPNoCap.fractionOfDaysOOS(SIStochasticPolicy, timeSteps, numTraces)
  
  logger.info(f"Deterministic Policy yields ${ossFractionDP * 100}%1.2f of Out-Of-Stock days")
  logger.info(f"Stochastic Policy yields ${ossFractionSP * 100}%1.2f of Out-Of-Stock days")
  
}