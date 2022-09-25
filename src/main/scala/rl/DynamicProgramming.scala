package rl

import breeze.linalg._
import breeze.numerics._
import com.typesafe.scalalogging.Logger
import rl.DynamicProgramming._
import rl.IterateUtils._
import rl.utils.{Categorical, Choose, FiniteDistribution}

import java.util.Locale

object DynamicProgramming {
  type ValueFunction[S] = Map[NonTerminal[S], Double]

  implicit val defaultTolerance: Double = 1.0e-5

  def policyIterationResult[S, A](
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    matrixMethodForMRPEval: Boolean = false
  ): (ValueFunction[S], FinitePolicy[S, A]) = {
    val res = converged(
      policyIteration[S, A](mdp, gamma, matrixMethodForMRPEval),
      almostEqualValueFunctionAndPolicy[S, A]
    )
    res
  }
  
  /**
   * Calculate the value function (V*) of the given MDP by improving
   * the policy repeatedly after evaluating the value function for a policy
   */
  def policyIteration[S, A](
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: Double,
    matrixMethodForMRPEval: Boolean = false
  ): Iterator[(ValueFunction[S], FinitePolicy[S, A])] = {
    
    def update(
      valueFunctionAndPolicy: (ValueFunction[S], FinitePolicy[S, A])
    ): (ValueFunction[S], FinitePolicy[S, A]) = {
      val (_, policy) = valueFunctionAndPolicy
      val mrp = mdp.applyFinitePolicy(policy)
      val policyValueFunction: ValueFunction[S] = if (matrixMethodForMRPEval) {
        val vFunc = mrp.valueFunctionVector(gamma)
        mergeIntoMap(vFunc.valuesIterator, mrp.nonTerminalStates)
      }
      else {
        evaluateMarkovRewardProcessResult(mrp, gamma)
      }
      
      val improvedPolicy = greedyPolicyFromValueFunction(mdp, policyValueFunction, gamma)
      (policyValueFunction, improvedPolicy)
    }
    
    val states = mdp.nonTerminalStates
    val v0: ValueFunction[S] = states.map { s => s -> 0.0 }.toMap
    val policy0: FinitePolicy[S, A] = new FinitePolicy[S, A] {
      override def policyMap: Map[S, FiniteDistribution[A]] = states.map { state =>
        state.state -> Choose(mdp.actions(state))
      }.toMap
    }
    
    iterate(update, (v0, policy0))
  }
  
  def evaluateMarkovRewardProcessResult[S](
    mrp: FiniteMarkovRewardProcess[S],
    gamma: Double
  ): ValueFunction[S] = {
    val vStar = converged(
      evaluateMarkovRewardProcess(mrp, gamma),
      done = almostEqualVectors
    )
    
    mergeIntoMap(vStar.valuesIterator, mrp.nonTerminalStates)
  }
  
  /**
   * Iteratively calculate the value function for the given Markov reward process.
   */
  def evaluateMarkovRewardProcess[S](
    mrp: FiniteMarkovRewardProcess[S],
    gamma: Double
  ): Iterator[DenseVector[Double]] = {
    def update(v: DenseVector[Double]): DenseVector[Double] = {
      mrp.rewardVector + gamma * mrp.transitionMatrix * v
    }
    
    val v0 = DenseVector.zeros[Double](mrp.nonTerminalStates.length)
    iterate(update, v0)
  }
  
  def almostEqualVectors(
    v1: DenseVector[Double],
    v2: DenseVector[Double])(
    implicit tolerance: Double = defaultTolerance
  ): Boolean = max(abs(v1 - v2)) < tolerance
  
  private def mergeIntoMap[A, B](
    iterator1: Iterator[A],
    iterable2: Iterable[B]
  ): Map[B, A] = {
    iterator1.zip(iterable2).map { case (a, b) => b -> a }.toMap
  }
  
  def greedyPolicyFromValueFunction[S, A](
    mdp: FiniteMarkovDecisionProcess[S, A],
    valueFunction: ValueFunction[S],
    gamma: Double
  ): FiniteDeterministicPolicy[S, A] = {

    implicit object ActionValueOrdering extends math.Ordering[(A, Double)] {
      override def compare(x: (A, Double), y: (A, Double)): Int = mdp.compareActionValueTuple(x, y)
    }

    val greedyPolicyMap: Map[S, A] = mdp.nonTerminalStates.map { state =>
      val qValues: Iterable[(A, Double)] = mdp.actions(state).map { action =>
        action -> mdp.stateActionMap(state)(action).expectation { case (nextState, reward) =>
          reward + gamma * extendedValueFunction(valueFunction)(nextState)
        }
      }

      val optimalAction: A = qValues.max match {
        case (a, _) => a
      }

      state.state -> optimalAction
    }.toMap
    
    FiniteDeterministicPolicy(greedyPolicyMap)
  }
  
  def extendedValueFunction[S](valueFunction: ValueFunction[S])(state: State[S]): Double = {
    state.onNonTerminal(valueFunction.apply, 0.0)
  }
  
  def almostEqualValueFunctionAndPolicy[S, A](
    valueFunctionAndPolicy1: (ValueFunction[S], FinitePolicy[S, A]),
    valueFunctionAndPolicy2: (ValueFunction[S], FinitePolicy[S, A])
  )(
    implicit tolerance: Double = defaultTolerance
  ): Boolean = {
    val (vf1, _) = valueFunctionAndPolicy1
    val (vf2, _) = valueFunctionAndPolicy2
    vf1.keys.map { state =>
      math.abs(vf1(state) - vf2(state))
    }.max < tolerance
  }
  
  def valueIterationResult[S, A](
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: Double
  ): (ValueFunction[S], FiniteDeterministicPolicy[S, A]) = {
    val optValueFunction = converged(valueIteration[S, A](mdp, gamma), almostEqualValueFunction[S])
    val policy = greedyPolicyFromValueFunction(mdp, optValueFunction, gamma)
    (optValueFunction, policy)
  }
  
  /**
   * Calculate the value function (V*) of the given MDP by applying the
   * update function repeatedly until the values converge.
   */
  def valueIteration[S, A](
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: Double
  ): Iterator[ValueFunction[S]] = {
    
    def update(
      valueFunction: ValueFunction[S]
    ): ValueFunction[S] = {
      val states = mdp.nonTerminalStates
      states.map { state =>
        state -> mdp.actions(state).map { action =>
          mdp.stateActionMap(state)(action).expectation { case (nextState, reward) =>
            reward + gamma * extendedValueFunction(valueFunction)(nextState)
          }
        }.max
      }.toMap
    }
    
    val states = mdp.nonTerminalStates
    val v0: ValueFunction[S] = states.map { s => s -> 0.0 }.toMap
    
    iterate(update, v0)
  }
  
  def almostEqualValueFunction[S](
    valueFunction1: ValueFunction[S],
    valueFunction2: ValueFunction[S]
  )(
    implicit tolerance: Double = defaultTolerance
  ): Boolean = {
    valueFunction1.keys.map { state =>
      math.abs(valueFunction1(state) - valueFunction2(state))
    }.max < tolerance
  }
  
  def valueFunctionToString[S](
    valueFunction: ValueFunction[S],
    optionalStateSortingFunction: Option[(NonTerminal[S], NonTerminal[S]) => Boolean] = None
  ): String = {
    val values = valueFunction.toSeq
    val sortedValues = optionalStateSortingFunction
      .map(f => values.sortWith {case ((x, _), (y, _)) => f(x, y)})
      .getOrElse(values)
    sortedValues.map { case (state, value) => f"Value for $state: $value%1.4f" }.mkString("\n")
  }
}

object DynamicProgrammingApp extends App {

  val logger: Logger = Logger("DynamicProgrammingApp")
  Locale.setDefault(Locale.US) // To print numbers in US format

  val rewardMap = Map(
    1 -> Categorical(Map((1, 7.0) -> 0.6, (2, 5.0) -> 0.3, (3, 2.0) -> 0.1)),
    2 -> Categorical(Map((1, -2.0) -> 0.1, (2, 4.0) -> 0.2, (3, 0.0) -> 0.7)),
    3 -> Categorical(Map((1, 3.0) -> 0.2, (2, 8.0) -> 0.6, (3, 4.0) -> 0.2))
  )
  val gamma: Double = 0.9

  val FMRP = FiniteMarkovRewardProcess(rewardMap)

  val stationaryDistribution = FMRP.stationaryDistribution

  logger.info(s"Stationary Distribution:\n$stationaryDistribution")

  val rewardVector = FMRP.rewardVector
  val rewardVectorStr = FMRP.rewardVectorToString
  val valueFunction = FMRP.valueFunctionVector(gamma)
  val valueFunctionStr = FMRP.valueFunctionToString(gamma)
  
  logger.info(s"Reward Vector:\n$rewardVectorStr")
  logger.info(s"Value Function (gamma = $gamma):\n$valueFunctionStr")
  
  val mrpResult = evaluateMarkovRewardProcessResult(FMRP, gamma)
  val mrpResultStr = valueFunctionToString(mrpResult)
  logger.info(s"Implied MRP Policy Iteration Value Function (gamma = $gamma):\n$valueFunctionStr")
}
