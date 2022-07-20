package rl.chapter12

import java.time.LocalDateTime
import java.util.Locale

import breeze.numerics._
import breeze.plot._
import com.typesafe.scalalogging.Logger
import rl.DynamicProgramming.ValueFunction
import rl.FiniteMarkovDecisionProcess.{StateActionMapping, processInputMap}
import rl.utils.{Categorical, Choose, ControlUtils, Utils}
import rl._

class VampireMDP(
  val initialVillagers: Int
) extends FiniteMarkovDecisionProcess[Int, Int] {
  
  override val stateActionMap: StateActionMapping[Int, Int] = {
    val inputMap = (1 to initialVillagers).map { i =>
      i -> (0 until i).map { a =>
        val prob = a.toDouble / i.toDouble
        a -> Categorical(
          Map(
            (i - a - 1, 0.0) -> (1.0 - prob),
            (0, (i - a).toDouble) -> prob
          )
        )
      }.toMap
    }.toMap
    
    processInputMap(inputMap)
  }
  
  override def stateSortingFunction(x: NonTerminal[Int], y: NonTerminal[Int]): Boolean = x.state <= y.state
  
  override def actionSortingFunction(x: Int, y: Int): Boolean = x <= y
  
  def valueIterationValueFunctionAndPolicy: (ValueFunction[Int], FiniteDeterministicPolicy[Int, Int]) =
    DynamicProgramming.valueIterationResult(this, 1.0)
  
  def lspiValueFunctionAndPolicy(
    numTransitions: Int = 5.0e4.toInt,
    numIterations: Int = 100
  ): (ValueFunction[Int], FiniteDeterministicPolicy[Int, Int]) = {
    val transitions = lspiTransitions.flatten.take(numTransitions)
    val initialTargetPolicy: DeterministicPolicy[Int, Int] = (state: Int) => state / 2
    val qValueFunctions = TemporalDifference.leastSquaresPolicyIteration(
      transitions = transitions,
      featureFunctions = lspiFeatureFunctions(4, 4),
      actions = this.actions,
      initialTargetPolicy = initialTargetPolicy,
      gamma = 1.0,
      epsilon = 1.0e-5
    )
    val finalQValue = qValueFunctions.drop(numIterations).next()
    ControlUtils.getValueFunctionAndPolicyFromQValueFunction(this, finalQValue)
  }
  
  def lspiFeatureFunctions(factor1Features: Int, factor2Features: Int): Seq[((NonTerminal[Int], Int)) => Double] = {
    val laguerre1 = Utils.laguerrePolynomials.take(factor1Features)
    val laguerre2 = Utils.laguerrePolynomials.take(factor2Features)
    
    def functor1(f: Double => Double)(x: (NonTerminal[Int], Int)): Double = {
      val (s, a) = x
      f(pow(s.state - a, 2) / s.state.toDouble)
    }
    
    def functor2(f: Double => Double)(x: (NonTerminal[Int], Int)): Double = {
      val (s, a) = x
      f((s.state - a).toDouble * a / s.state.toDouble)
    }
    
    val factor1Functions: Seq[((NonTerminal[Int], Int)) => Double] = laguerre1.map { f => functor1(f) }
    val factor2Functions: Seq[((NonTerminal[Int], Int)) => Double] = laguerre2.map { f => functor2(f) }
    factor1Functions ++ factor2Functions
  }
  
  def lspiTransitions: LazyList[LazyList[ActionStep[Int, Int]]] = {
    val initialDistribution = Choose(nonTerminalStates)
    
    def policy: UniformPolicy[Int, Int] = (state: Int) => 0 until state
    
    this.actionTraces(initialDistribution, policy)
  }
  
}

object VampireMDP {
  
  def apply(initialVillagers: Int): VampireMDP = new VampireMDP(initialVillagers)
  
}

object VampireMDPApp extends App {
  
  val logger: Logger = Logger("VampireMDPApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val initialVillagers = 20
  val vampireMDP = VampireMDP(initialVillagers)
  val nonTerminalStates = vampireMDP.nonTerminalStates
  
  val (trueValueFunction, truePolicy) = vampireMDP.valueIterationValueFunctionAndPolicy
  val sortingFunction: (NonTerminal[Int], NonTerminal[Int]) => Boolean = vampireMDP.stateSortingFunction
  val trueValueFunctionStr = DynamicProgramming.valueFunctionToString(trueValueFunction, Some(sortingFunction))
  val truePolicyStr = truePolicy.printPolicy(_ <= _)
  
  val numTransitions = 5.0e4.toInt
  val numIterations = 100
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val (lspiValueFunction, lspiPolicy) = vampireMDP.lspiValueFunctionAndPolicy(numTransitions, numIterations)
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  val lspiValueFunctionStr = DynamicProgramming.valueFunctionToString(lspiValueFunction, Some(sortingFunction))
  val lspiPolicyStr = lspiPolicy.printPolicy(_ <= _)
  
  val xs = nonTerminalStates.map(_.state.toDouble)
  val trueValues = nonTerminalStates.map(trueValueFunction.apply)
  val trueActions = nonTerminalStates.map(nt => truePolicy.actionForState(nt.state).toDouble)
  val lspiValues = nonTerminalStates.map(lspiValueFunction.apply)
  val lspiActions = nonTerminalStates.map(nt => lspiPolicy.actionForState(nt.state).toDouble)
  
  logger.info(f"True Value Function:\n$trueValueFunctionStr")
  logger.info(f"True Policy:\n$truePolicyStr")
  logger.info(f"LSPI Value Function:\n$lspiValueFunctionStr")
  logger.info(f"LSPI Policy:\n$lspiPolicyStr")
  
  val fig = Figure("True Optimal VF versus LSPI-Estimated Optimal VF")
  val p = fig.subplot(0)
  p += plot(xs, trueValues, name = "True Value Function")
  p += plot(xs, lspiValues, name = "LSPI Value Function")
  p.legend = true
  
  val fig2 = Figure("True Optimal Policy versus LSPI-Estimated Optimal Policy")
  val p2 = fig2.subplot(0)
  p2 += plot(xs, trueActions, name = "True Policy")
  p2 += plot(xs, lspiActions, name = "LSPI Policy")
  p2.legend = true
  
}
