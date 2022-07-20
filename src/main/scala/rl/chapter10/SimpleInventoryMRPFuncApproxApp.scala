package rl.chapter10

import java.time.LocalDateTime
import java.util.Locale

import breeze.linalg._
import breeze.numerics._
import breeze.plot._
import breeze.stats._
import com.typesafe.scalalogging.Logger
import rl.ApproximateDynamicProgramming.ValueFunctionApproximation
import rl.chapter2.{InventoryState, SimpleInventoryMRPFinite}
import rl.utils.{Choose, PredictionUtils}
import rl.{AdamGradient, LinearFunctionApproximation, NonTerminal}

object SimpleInventoryMRPFuncApproxApp extends App {
  
  val logger: Logger = Logger("SimpleInventoryMRPFuncApprox")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val capacity = 2
  val poissonLambda = 1.0
  val holdingCost = 1.0
  val stockoutCost = 10.0
  
  val gamma = 0.9
  
  val SIMRPFinite = SimpleInventoryMRPFinite(capacity, poissonLambda, holdingCost, stockoutCost)
  
  val nonTerminalStates = SIMRPFinite.nonTerminalStates
  val trueValueFunction = SIMRPFinite.valueFunctionVector(gamma)
  val trueValueFunctionStr = SIMRPFinite.valueFunctionToString(gamma)
  
  val episodeLengthTolerance = 1.0e-6
  val episodeLength = 200
  val halfLife = 1.0e4
  val exponent = 0.5
  
  val initialLearningRate = 0.03
  
  val lambda = 0.3
  
  val featureFunctions: Seq[NonTerminal[InventoryState] => Double] = nonTerminalStates.map { s =>
    (x: NonTerminal[InventoryState]) => if (x.state == s.state) 1.0 else 0.0
  }
  
  val mcAdamGradient = AdamGradient(
    learningRate = 0.05,
    decay1 = 0.9,
    decay2 = 0.999
  )
  
  val tdAdamGradient = AdamGradient(
    learningRate = 0.05,
    decay1 = 0.9,
    decay2 = 0.999
  )
  
  val mcFuncApprox: LinearFunctionApproximation[NonTerminal[InventoryState]] =
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      adamGradient = mcAdamGradient
    )
    
  val tdFuncApprox: LinearFunctionApproximation[NonTerminal[InventoryState]] =
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      adamGradient = tdAdamGradient
    )
  
  val mcEpisodes: Int = 3000
  val tdExperiences: Int = 30000
  val tdLambdaEpisodes: Int = 30000
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val mcValueFunctions: Iterable[ValueFunctionApproximation[InventoryState]] =
    PredictionUtils.mcPredictionLearningRate(
      markovRewardProcess = SIMRPFinite,
      initialStateDistribution = Choose(nonTerminalStates),
      gamma = gamma,
      initialApproximation = mcFuncApprox,
      episodeLengthTolerance = episodeLengthTolerance
    ).take(mcEpisodes).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val tdValueFunctions: Iterable[ValueFunctionApproximation[InventoryState]] =
    PredictionUtils.tdPredictionLearningRate(
      markovRewardProcess = SIMRPFinite,
      initialStateDistribution = Choose(nonTerminalStates),
      gamma = gamma,
      initialApproximation = tdFuncApprox,
      episodeLength = episodeLength
    ).take(tdExperiences).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val tdLambdaValueFunctions: Iterable[ValueFunctionApproximation[InventoryState]] =
    PredictionUtils.tdLambdaPredictionLearningRate(
      markovRewardProcess = SIMRPFinite,
      initialStateDistribution = Choose(nonTerminalStates),
      gamma = gamma,
      lambda = lambda,
      initialApproximation = tdFuncApprox,
      episodeLength = episodeLength
    ).take(tdLambdaEpisodes).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val mcFiniteValueFunctions: Iterable[ValueFunctionApproximation[InventoryState]] =
    PredictionUtils.mcFinitePredictionLearningRate(
      finiteMarkovRewardProcess = SIMRPFinite,
      gamma = gamma,
      episodeLengthTolerance = episodeLengthTolerance,
      initialLearningRate = initialLearningRate,
      halfLife = halfLife,
      exponent = exponent
    ).take(mcEpisodes).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val tdFiniteValueFunctions: Iterable[ValueFunctionApproximation[InventoryState]] =
    PredictionUtils.tdFinitePredictionLearningRate(
      finiteMarkovRewardProcess = SIMRPFinite,
      gamma = gamma,
      episodeLength = episodeLength,
      initialLearningRate = initialLearningRate,
      halfLife = halfLife,
      exponent = exponent
    ).take(tdExperiences).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val tdLambdaFiniteValueFunctions: Iterable[ValueFunctionApproximation[InventoryState]] =
    PredictionUtils.tdLambdaFinitePredictionLearningRate(
      finiteMarkovRewardProcess = SIMRPFinite,
      gamma = gamma,
      lambda = lambda,
      episodeLength = episodeLength,
      initialLearningRate = initialLearningRate,
      halfLife = halfLife,
      exponent = exponent
    ).take(tdLambdaEpisodes).toSeq
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(f"MC Iteration:")
  val (stepsMC, errorsMC) = mcValueFunctions
    .zipWithIndex
    .map { case (vf, i) =>
      val pred: DenseVector[Double] = vf.evaluate(nonTerminalStates)
      val error = pred - trueValueFunction
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 50 == 0) {
        logger.info(f"MC Iteration $i: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == mcEpisodes) {
        val mcVFStr = nonTerminalStates
          .map{s => f"Value for $s: ${vf(s)}%1.4f"}
          .mkString("\n")
        logger.info(f"MC Value Function:\n$mcVFStr")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  logger.info(f"MC Finite Iteration:")
  val (stepsMCFinite, errorsMCFinite) = mcFiniteValueFunctions
    .zipWithIndex
    .map { case (vf, i) =>
      val pred: DenseVector[Double] = vf.evaluate(nonTerminalStates)
      val error = pred - trueValueFunction
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 50 == 0) {
        logger.info(f"MC Finite Iteration $i: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == mcEpisodes) {
        val mcVFStr = nonTerminalStates
          .map{s => f"Value for $s: ${vf(s)}%1.4f"}
          .mkString("\n")
        logger.info(f"MC Finite Value Function:\n$mcVFStr")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  logger.info(f"TD Iteration:")
  val (stepsTD, errorsTD) = tdValueFunctions
    .zipWithIndex
    .map { case (vf, i) =>
      val pred: DenseVector[Double] = vf.evaluate(nonTerminalStates)
      val error = pred - trueValueFunction
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 50000 == 0) {
        logger.info(f"TD Iteration ${i+1}: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == tdExperiences) {
        val tdVFStr = nonTerminalStates
          .map{s => f"Value for $s: ${vf(s)}%1.4f"}
          .mkString("\n")
        logger.info(f"TD Value Function:\n$tdVFStr")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  logger.info(f"TD Finite Iteration:")
  val (stepsTDFinite, errorsTDFinite) = tdFiniteValueFunctions
    .zipWithIndex
    .map { case (vf, i) =>
      val pred: DenseVector[Double] = vf.evaluate(nonTerminalStates)
      val error = pred - trueValueFunction
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 50000 == 0) {
        logger.info(f"TD Finite Iteration ${i+1}: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == tdExperiences) {
        val tdVFStr = nonTerminalStates
          .map{s => f"Value for $s: ${vf(s)}%1.4f"}
          .mkString("\n")
        logger.info(f"TD Finite Value Function:\n$tdVFStr")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  logger.info(f"TD-Lambda Iteration:")
  val (stepsTDLambda, errorsTDLambda) = tdLambdaValueFunctions
    .zipWithIndex
    .map { case (vf, i) =>
      val pred: DenseVector[Double] = vf.evaluate(nonTerminalStates)
      val error = pred - trueValueFunction
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 1000 == 0) {
        logger.info(f"TD-Lambda Iteration ${i+1}: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == tdExperiences) {
        val tdVFStr = nonTerminalStates
          .map{s => f"Value for $s: ${vf(s)}%1.4f"}
          .mkString("\n")
        logger.info(f"TD-Lambda Value Function:\n$tdVFStr")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  logger.info(f"TD-Lambda Finite Iteration:")
  val (stepsTDLambdaFinite, errorsTDLambdaFinite) = tdLambdaFiniteValueFunctions
    .zipWithIndex
    .map { case (vf, i) =>
      val pred: DenseVector[Double] = vf.evaluate(nonTerminalStates)
      val error = pred - trueValueFunction
      val rmse = sqrt(mean(pow(error, 2)))
      if ((i + 1) % 1000 == 0) {
        logger.info(f"TD-Lambda Finite Iteration ${i+1}: RMSE: $rmse%1.4f")
      }
      if ((i + 1) == tdExperiences) {
        val tdVFStr = nonTerminalStates
          .map{s => f"Value for $s: ${vf(s)}%1.4f"}
          .mkString("\n")
        logger.info(f"TD-Lambda Finite Value Function:\n$tdVFStr")
        logger.info(f"True Value Function:\n$trueValueFunctionStr")
      }
      (i.toDouble, rmse)
    }
    .toIndexedSeq // to force conversion from LazyList
    .unzip
  
  val fig = Figure("MC Value Function Approximation")
  val p = fig.subplot(0)
  p += plot(stepsMC, errorsMC, name = "Monte-Carlo RMSE")
  p += plot(stepsMCFinite, errorsMCFinite, name = "Monte-Carlo Finite RMSE")
  p.legend = true
  
  val fig2 = Figure("TD Value Function Approximation")
  val p2 = fig2.subplot(0)
  p2 += plot(stepsTD, errorsTD, name = "TD RMSE")
  p2 += plot(stepsTDFinite, errorsTDFinite, name = "TD Finite RMSE")
  p2.legend = true
  
  val fig3 = Figure("TD-Lambda Value Function Approximation")
  val p3 = fig3.subplot(0)
  p3 += plot(stepsTDLambda, errorsTDLambda, name = "TD-Lambda RMSE")
  p3 += plot(stepsTDLambdaFinite, errorsTDLambdaFinite, name = "TD-Lambda Finite RMSE")
  p3.legend = true
  
}
