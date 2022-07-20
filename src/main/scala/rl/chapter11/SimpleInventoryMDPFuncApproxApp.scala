package rl.chapter11

import java.time.LocalDateTime
import java.util.Locale

import breeze.linalg._
import breeze.numerics._
import breeze.plot._
import breeze.stats._
import com.typesafe.scalalogging.Logger
import rl.DynamicProgramming.valueIterationResult
import rl.chapter2.InventoryState
import rl.{AdamGradient, LinearFunctionApproximation, NonTerminal}
import rl.chapter3.SimpleInventoryMDPCap
import rl.utils.ControlUtils.getValueFunctionAndPolicyFromQValueFunction
import rl.utils.{Choose, ControlUtils}

object SimpleInventoryMDPFuncApproxApp extends App {
  
  val logger: Logger = Logger("SimpleInventoryMDPApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val capacity = 2
  val poissonLambda = 1.0
  val holdingCost = 1.0
  val stockoutCost = 10.0
  
  val gamma = 0.9
  
  val SIMDPCap = SimpleInventoryMDPCap(capacity, poissonLambda, holdingCost, stockoutCost)
  val (trueValueFunction, truePolicy) = valueIterationResult(SIMDPCap, gamma)
  val nonTerminalStates: Seq[NonTerminal[InventoryState]] = SIMDPCap.nonTerminalStates
  val trueValues = DenseVector.apply(nonTerminalStates.toArray).map(trueValueFunction.apply)
  
  val episodeLengthTolerance = 1.0e-5
  val mcNumEpisodes = 10000
  val scale = 0.5
  val tdExperiences = 10000
  val sarsaEpisodeLength = 200
  val qLearningEpisodeLength = 500
  val initialLearningRate = 0.2
  val halfLife = 1.0e4
  val exponent = 1.0
  val qLearningEpsilon = 0.2
  val epsilonFunction: Int => Double = (k: Int) => pow(k, -0.5)
  
  val lambda = 0.3
  val mcAdamGradient = AdamGradient(
    learningRate = 0.05,
    decay1 = 0.9,
    decay2 = 0.999
  )
  val tdAdamGradient = AdamGradient(
    learningRate = 0.003,
    decay1 = 0.9,
    decay2 = 0.999
  )
  
  val featureFunctions: Seq[((NonTerminal[InventoryState], Int)) => Double] = nonTerminalStates.flatMap { s =>
    SIMDPCap.actions(s).map { action =>
      x: (NonTerminal[InventoryState], Int) => if ((x._1.state == s.state) && (x._2 == action)) 1.0 else 0.0
    }
  }
  val mcFuncApprox: LinearFunctionApproximation[(NonTerminal[InventoryState], Int)] =
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      adamGradient = mcAdamGradient
    )
  
  val tdFuncApprox: LinearFunctionApproximation[(NonTerminal[InventoryState], Int)] =
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      adamGradient = tdAdamGradient
    )
  
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val mcValueFunctions = ControlUtils.glieMCControlLearningRate(
    markovDecisionProcess = SIMDPCap,
    initialStateDistribution = Choose(nonTerminalStates),
    gamma = gamma,
    initialApproximation = mcFuncApprox,
    episodeLengthTolerance = episodeLengthTolerance,
    epsilonFunction = epsilonFunction
  ).take(mcNumEpisodes).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val sarsaValueFunctions = ControlUtils.glieSarsaLearningRate(
    markovDecisionProcess = SIMDPCap,
    initialStateDistribution = Choose(nonTerminalStates),
    gamma = gamma,
    initialApproximation = mcFuncApprox,
    maxEpisodeLength = sarsaEpisodeLength,
    epsilonFunction = epsilonFunction
  ).take(tdExperiences).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val qLearningValueFunctions = ControlUtils.qLearningLearningRate(
    markovDecisionProcess = SIMDPCap,
    initialStateDistribution = Choose(nonTerminalStates),
    gamma = gamma,
    initialApproximation = mcFuncApprox,
    epsilon = qLearningEpsilon,
    maxEpisodeLength = qLearningEpisodeLength
  ).take(tdExperiences).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(f"MC Iteration:")
  val (stepsMC, errorsMC) = mcValueFunctions
    .zipWithIndex
    .map { case (qVF, i) =>
      val (valueFunction, policy) =
        getValueFunctionAndPolicyFromQValueFunction(SIMDPCap, qVF)
      
      val pred = DenseVector.apply(nonTerminalStates.toArray).map(valueFunction.apply)
      val error = pred - trueValues
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 1000 == 0) {
        logger.info(f"GLIE-MC Iteration ${i + 1}: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == mcValueFunctions.length) {
        val valueFunctionStr = nonTerminalStates
          .map { s => f"Value for $s: ${valueFunction(s)}%1.4f" }
          .mkString("\n")
        
        val trueValueFunctionStr = nonTerminalStates
          .map { s => f"Value for $s: ${trueValueFunction(s)}%1.4f" }
          .mkString("\n")
        
        logger.info(f"GLIE-MC Value Function:\n$valueFunctionStr")
        logger.info(f"GLIE-MC Optimal Policy:\n$policy")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  logger.info(f"TD-SARSA Iteration:")
  val (stepsSarsa, errorsSarsa) = sarsaValueFunctions
    .zipWithIndex
    .map { case (qVF, i) =>
      val (valueFunction, policy) =
        getValueFunctionAndPolicyFromQValueFunction(SIMDPCap, qVF)
      
      val pred = DenseVector.apply(nonTerminalStates.toArray).map(valueFunction.apply)
      val error = pred - trueValues
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 1000 == 0) {
        logger.info(f"GLIE-SARSA Iteration ${i + 1}: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == sarsaValueFunctions.length) {
        val valueFunctionStr = nonTerminalStates
          .map { s => f"Value for $s: ${valueFunction(s)}%1.4f" }
          .mkString("\n")
        
        val trueValueFunctionStr = nonTerminalStates
          .map { s => f"Value for $s: ${trueValueFunction(s)}%1.4f" }
          .mkString("\n")
        
        logger.info(f"GLIE-SARSA Value Function:\n$valueFunctionStr")
        logger.info(f"GLIE-SARSA Optimal Policy:\n$policy")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  logger.info(f"TD-SARSA Iteration:")
  val (stepsQLearning, errorsQLearning) = qLearningValueFunctions
    .zipWithIndex
    .map { case (qVF, i) =>
      val (valueFunction, policy) =
        getValueFunctionAndPolicyFromQValueFunction(SIMDPCap, qVF)
      
      val pred = DenseVector.apply(nonTerminalStates.toArray).map(valueFunction.apply)
      val error = pred - trueValues
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 1000 == 0) {
        logger.info(f"Q-Learning Iteration ${i + 1}: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == sarsaValueFunctions.length) {
        val valueFunctionStr = nonTerminalStates
          .map { s => f"Value for $s: ${valueFunction(s)}%1.4f" }
          .mkString("\n")
        
        val trueValueFunctionStr = nonTerminalStates
          .map { s => f"Value for $s: ${trueValueFunction(s)}%1.4f" }
          .mkString("\n")
        
        logger.info(f"Q-Learning Value Function:\n$valueFunctionStr")
        logger.info(f"Q-Learning Optimal Policy:\n$policy")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  val fig = Figure("Value Function Approximation")
  val p = fig.subplot(0)
  p += plot(stepsMC, errorsMC, name = "GLIE Monte-Carlo RMSE")
  p.legend = true
  
  val fig2 = Figure("GLIE-SARSA Value Function Approximation")
  val p2 = fig2.subplot(0)
  p2 += plot(stepsSarsa, errorsSarsa, name = "GLIE-SARSA RMSE")
  p2 += plot(stepsQLearning, errorsQLearning, name = "Q-Learning RMSE")
  p2.legend = true
  
//  val fig3 = Figure("Q-Learning Value Function Approximation")
//  val p3 = fig3.subplot(0)
//  p3 += plot(stepsQLearning, errorsQLearning, name = "Q-Learning RMSE")
//  p2.legend = true
  
}
