package rl.chapter10

import java.time.LocalDateTime
import java.util.Locale

import com.typesafe.scalalogging.Logger
import rl.chapter2.SimpleInventoryMRPFinite
import rl.utils.PredictionUtils

object SimpleInventoryMRPApp extends App {
  
  val logger: Logger = Logger("SimpleInventoryMRPFuncApprox")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val capacity = 2
  val poissonLambda = 1.0
  val holdingCost = 1.0
  val stockoutCost = 10.0
  
  val gamma = 0.9
  
  val SIMRPFinite = SimpleInventoryMRPFinite(capacity, poissonLambda, holdingCost, stockoutCost)
  
  val nonTerminalStates = SIMRPFinite.nonTerminalStates
  
  val episodeLengthTolerance = 1.0e-6
  val numEpisodes = 10000
  val episodeLength = 100
  val initialLearningRate = 0.03
  val halfLife = 1000.0
  val exponent = 0.5
  
  val lambda = 0.3
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  PredictionUtils.mcFiniteEqualWeightsCorrectness(
    finiteMarkovRewardProcess = SIMRPFinite,
    gamma = gamma,
    episodeLengthTolerance = episodeLengthTolerance,
    numEpisodes = numEpisodes
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  PredictionUtils.mcFinitePredictionLearningRateCorrectness(
    finiteMarkovRewardProcess = SIMRPFinite,
    gamma = gamma,
    episodeLengthTolerance = episodeLengthTolerance,
    numEpisodes = numEpisodes,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  PredictionUtils.tdFiniteLearningRateCorrectness(
    finiteMarkovRewardProcess = SIMRPFinite,
    gamma = gamma,
    episodeLength = episodeLength,
    numEpisodes = numEpisodes * episodeLength,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  PredictionUtils.tdLambdaFiniteLearningRateCorrectness(
    finiteMarkovRewardProcess = SIMRPFinite,
    gamma = gamma,
    lambda = lambda,
    episodeLength = episodeLength,
    numEpisodes = numEpisodes,
    initialLearningRate = initialLearningRate,
    halfLife = halfLife,
    exponent = exponent
  )
  logger.info(s"Finished computation at ${LocalDateTime.now()}")
  
}
