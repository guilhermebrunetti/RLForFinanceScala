package rl

import breeze.linalg._
import breeze.linalg.eig.DenseEig
import breeze.numerics._
import rl.FiniteMarkovProcess.{Transition, defaultSortingFunction}
import rl.utils.{Categorical, Distribution, FiniteDistribution}

sealed trait State[S] {
  def state: S
  
  def onNonTerminal[B](f: NonTerminal[S] => B, defaultValue: B): B =
    this.onNonTerminalOption(f).getOrElse(defaultValue)
  
  def onNonTerminalOption[B](f: NonTerminal[S] => B): Option[B] = this match {
    case nt: NonTerminal[S] => Some(f(nt))
    case _: Terminal[S] => None
  }
  
  def map[B](f: S => B): State[B] = this match {
    case nt: NonTerminal[S] => nt.map(f)
    case t: Terminal[S] => t.map(f)
  }
  
}

case class NonTerminal[S](state: S) extends State[S] {
  override def map[B](f: S => B): NonTerminal[B] = NonTerminal(f(state))
}

case class Terminal[S](state: S) extends State[S] {
  override def map[B](f: S => B): State[B] = Terminal(f(state))
}

trait MarkovProcess[S] {
  
  /**
   * Given a state of process, returns a distribution of the next state
   */
  def transition(state: NonTerminal[S]): Distribution[State[S]]
  
  def traces(initialStateDistribution: Distribution[NonTerminal[S]]): LazyList[LazyList[State[S]]] = {
    LazyList.continually(simulate(initialStateDistribution))
  }
  
  /**
   * Run a simulation trace of this Markov process
   */
  def simulate(initialStateDistribution: Distribution[NonTerminal[S]]): LazyList[State[S]] = {
    val initialState = initialStateDistribution.sample
    simulate(initialState)
  }
  
  def simulate(initialState: State[S]): LazyList[State[S]] = {
    LazyList.unfold(Option(initialState))(_.map(nextStep))
  }
  
  private def nextStep(step: State[S]): (State[S], Option[State[S]]) = {
    (step, step.onNonTerminalOption(transition(_).sample))
  }
  
}

/**
 * A Markov process with finite state space
 * Having finite state lets us to use tabular methods to work
 * with the process (i.e. dynamic programming).
 */
trait FiniteMarkovProcess[S]
  extends MarkovProcess[S] {
  
  def transitionMap: Transition[S]
  
  override def toString: String = {
    transitionMap
      .toIndexedSeq
      .sortWith { case ((x, _), (y, _)) => sortingFunction(x, y) }
      .mkString("\n")
  }
  
  def sortingFunction(x: NonTerminal[S], y: NonTerminal[S]): Boolean = defaultSortingFunction(x, y)
  
  def stationaryDistribution: FiniteDistribution[S] = {
    val denseEigenSystem: DenseEig = eig(transitionMatrix.t)
    val eigenValues: DenseVector[Double] = denseEigenSystem.eigenvalues
    val eigenVectors: DenseMatrix[Double] = denseEigenSystem.eigenvectors
    val firstUnitEigenValueIndex: Int = where(abs(eigenValues - 1.0) <:< 1.0e-8).head
    val firstUnitEigenVector = eigenVectors(::, firstUnitEigenValueIndex)
    val total = sum(firstUnitEigenVector)
    val massMap: Map[S, Double] = firstUnitEigenVector
      .valuesIterator
      .zip(nonTerminalStates).map {
      case (m, s) => s.state -> m / total
    }.toMap
    Categorical(massMap)
  }
  
  def transitionMatrix: DenseMatrix[Double] = {
    val rows = nonTerminalStates.map { s1 =>
      nonTerminalStates.map(s2 => this.transition(s1).probability(s2))
    }
    
    DenseMatrix(rows: _*)
  }
  
  def nonTerminalStates: IndexedSeq[NonTerminal[S]] = transitionMap.keySet.toIndexedSeq.sortWith(sortingFunction)
  
  override def transition(state: NonTerminal[S]): FiniteDistribution[State[S]] = {
    transitionMap(state)
  }
  
}

object FiniteMarkovProcess {
  type Transition[S] = Map[NonTerminal[S], FiniteDistribution[State[S]]]
  
  def defaultSortingFunction[X](x: X, y: X) = true
  
  def transitionsFromMap[S](transitionMap: Map[S, FiniteDistribution[S]]): Transition[S] = {
    val nonTerminalStates = transitionMap.keySet
    
    def toState(s: S): State[S] = if (nonTerminalStates.contains(s)) NonTerminal(s) else Terminal(s)
    
    transitionMap.map { case (s, fd) => NonTerminal(s) -> fd.map(toState) }
  }
}

