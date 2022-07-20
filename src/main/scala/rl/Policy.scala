package rl

import rl.FiniteMarkovProcess.defaultSortingFunction
import rl.utils.{Choose, Constant, Distribution, FiniteDistribution}

trait Policy[S, A] {
  def act(state: NonTerminal[S]): Distribution[A]
}

/**
 * A policy that randomly selects one of several specified policies
 * each action.
 *
 * Given the right inputs, this could simulate things like ε-greedy
 * policies
 */
trait RandomPolicy[S, A] extends Policy[S, A] {
  
  def policyChoices: Distribution[Policy[S, A]]
  
  override def act(state: NonTerminal[S]): Distribution[A] = {
    val policy = policyChoices.sample
    policy.act(state)
  }
}

trait UniformPolicy[S, A] extends Policy[S, A] {
  
  def validAction(state: S): Iterable[A]
  
  override def act(state: NonTerminal[S]): Distribution[A] = {
    Choose(validAction(state.state))
  }
}

trait DeterministicPolicy[S, A] extends Policy[S, A] {
  
  def actionForState(state: S): A
  
  override def act(state: NonTerminal[S]): Constant[A] = {
    Constant(actionForState(state.state))
  }
}

class Always[S, A](
  val constantAction: A
) extends DeterministicPolicy[S, A] {
  
  override def actionForState(state: S): A = constantAction
}

trait FinitePolicy[S, A] extends Policy[S, A] {
  
  def policyMap: Map[S, FiniteDistribution[A]]
  
  override def act(state: NonTerminal[S]): FiniteDistribution[A] = policyMap(state.state)
}

class FiniteDeterministicPolicy[S, A](
  val actionMap: Map[S, A]
) extends FinitePolicy[S, A] with DeterministicPolicy[S, A] {
  
  override def policyMap: Map[S, Constant[A]] = {
    actionMap.view.mapValues(Constant(_)).toMap
  }
  
  override def actionForState(state: S): A = actionMap(state)
  
  override def toString: String = {
    printPolicy(defaultSortingFunction)
  }
  
  def printPolicy(sortingFunction: (S, S) => Boolean): String = {
    actionMap
      .toSeq
      .sortWith {case ((x, _), (y, _)) => sortingFunction(x, y)}
      .map { case (s, a) => s"For State $s: Do action $a" }
      .mkString("\n")
  }
}

object FiniteDeterministicPolicy {
  
  def apply[S, A](actionMap: Map[S, A]): FiniteDeterministicPolicy[S, A] =
    new FiniteDeterministicPolicy(actionMap)
}