package rl.chapter12

import java.util.Locale

import breeze.plot._
import com.typesafe.scalalogging.Logger
import rl.Tabular.learningRateSchedule
import rl.chapter10.RandomWalkMRP
import rl.utils.{Choose, Utils}
import rl.{NonTerminal, Tabular, TemporalDifference, TransitionStep}

object RandomWalkLSTDApp extends App {
  
  val logger: Logger = Logger("RandomWalkLSTDApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val barrier: Int = 20
  val probability: Double = 0.55
  val randomWalkMRP = RandomWalkMRP(barrier, probability)
  
  val initialLearningRate = 0.5
  val halfLife = 1.0e3
  val exponent = 0.5
  val gamma = 1.0
  val lambda = 0.3
  
  val nonTerminalStates = randomWalkMRP.nonTerminalStates
  val trueValueFunction = randomWalkMRP.valueFunctionVector(gamma)
  val trueValueFunctionStr = randomWalkMRP.valueFunctionToString(gamma)
  
  val numTransitions = 1.0e5.toInt
  
  val initialStateDistribution = Choose(nonTerminalStates)
  val traces: LazyList[LazyList[TransitionStep[Int]]] = randomWalkMRP.rewardTraces(initialStateDistribution)
  val transitions: LazyList[TransitionStep[Int]] = traces.flatten
  val tdTransitions = transitions.take(numTransitions).toIndexedSeq // force initialization
  
  val learningRateFunction: Int => Double = learningRateSchedule(initialLearningRate, halfLife, exponent)(_)
  val initialApproximation = Tabular[NonTerminal[Int]](
    countToWeight = learningRateFunction
  )
  
  val tdFunction = TemporalDifference.tdPrediction(
    transitions = tdTransitions,
    initialApproximation = initialApproximation
  ).toSeq.last
  
  val laguerrePolynomials: Seq[Double => Double] = Utils.laguerrePolynomials.take(5)
  
  val featureFunctions: Seq[NonTerminal[Int] => Double] = laguerrePolynomials.map { f =>
    (x: NonTerminal[Int]) => f(x.state)
  }
  
  val epsilon = 1.0e-4
  val lstdValueFunction = TemporalDifference.leastSquaresTD(
    transitions = tdTransitions,
    featureFunctions = featureFunctions,
    gamma = gamma,
    epsilon = epsilon
  )
  
  val xs = nonTerminalStates.map(_.state.toDouble)
  val tdValues = tdFunction.evaluate(nonTerminalStates)
  val lstdValues = lstdValueFunction.evaluate(nonTerminalStates)
  
  val fig = Figure("Tabular TD and Least-Squares TD versus True Value Function")
  val p = fig.subplot(0)
  p += plot(xs, trueValueFunction, name = "True Value Function")
  p += plot(xs, tdValues, name = "Tabular TD Value Function")
  p += plot(xs, lstdValues, name = "Least-Squares TD Value Function")
  p.legend = true
  
}
