package rl.chapter5

import java.util.Locale

import com.typesafe.scalalogging.Logger
import rl.chapter5.FunctionApproximationSimpleExamples._
import rl.utils.Gaussian
import rl.{AdamGradient, DNNApproximation, DNNSpec, LinearFunctionApproximation}
import breeze.linalg._

object FunctionApproximationSimpleExamples {
  
  type Triple = (Double, Double, Double)
  type DataSeq = Seq[(Triple, Double)]
  
  val dataGeneratorDistribution: Gaussian = Gaussian()
  val noiseDistribution: Gaussian = Gaussian(sigma = 0.3)
  val featureFunctions: Seq[Triple => Double] = Seq(
    _ => 1.0,
    x => x._1,
    x => x._2,
    x => x._3
  )
  
  def exampleModelDataGenerator: LazyList[(Triple, Double)] = {
    
    val coefficients = (2.0, 10.0, 4.0, -6.0)
    val (c0, c1, c2, c3) = coefficients
    
    def iterate: LazyList[(Triple, Double)] = {
      val (x1, x2, x3) = dataGeneratorDistribution.samples(3) match {
        case Seq(a, b, c) => (a, b, c)
      }
      val y = c0 + c1 * x1 + c2 * x2 + c3 * x3 + noiseDistribution.sample
      val xy = (x1, x2, x3) -> y
      xy #:: iterate
    }
    
    iterate
  }
  
  def dataSeqGenerator(
    dataGenerator: LazyList[(Triple, Double)],
    numberOfPoints: Int
  ): LazyList[DataSeq] = {
    LazyList.continually(dataGenerator.take(numberOfPoints))
  }
  
  def getLinearModel: LinearFunctionApproximation[Triple] = {
    
    LinearFunctionApproximation.create(
      featureFunctions = featureFunctions,
      adamGradient = adamGradient)
  }
  
  def adamGradient: AdamGradient = AdamGradient(learningRate = 0.1)
  
  def getDNNModel: DNNApproximation[Triple] = {
    def relu(x: Double): Double = if (x > 0) x else 0.0
    def relu_deriv(x: Double): Double = if (x > 0) 1.0 else 0.0
    def identity(x: Double): Double = x
    def identity_deriv(x: Double): Double = 1.0
    
    val dnnSpec = DNNSpec(
      neurons = Seq(2),
      bias = true,
      hiddenActivation = relu,
      hiddenActivationDerivative = relu_deriv,
      outputActivation = identity,
      outputActivationDerivative = identity_deriv
    )
    
    DNNApproximation.create(
      featureFunctions = featureFunctions,
      dnnSpec = dnnSpec,
      adamGradient = adamGradient,
      regularizationCoefficient = 0.05
    )
  }
}

object FunctionApproximationSimpleExamplesApp extends App {
  val logger: Logger = Logger("FunctionApproximationSimpleExamplesApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val trainNumberOfPoints: Int = 1000
  val trainIterations: Int = 300
  val testNumberOfPoints: Int = 10000
  
  val trainDataGenerator: LazyList[(Triple, Double)] = exampleModelDataGenerator
  val testDataGenerator: LazyList[(Triple, Double)] = exampleModelDataGenerator
  val trainDataStream = dataSeqGenerator(trainDataGenerator, trainNumberOfPoints)
  val testData = testDataGenerator.take(testNumberOfPoints)
  
  val firstIteration = getDNNModel.forwardPropagation(trainDataStream.head.map(_._1).take(10))
  logger.info(f"Initial Forward Propagation = \n${firstIteration.mkString("\n")}\n-----------------------------")
  
  val dnnModels = getDNNModel.iterateUpdates(trainDataStream).take(trainIterations + 1)
  
  dnnModels
    .zipWithIndex
    .foreach {
      case (f, i) =>
        val testError = f.rmse(testData)
        logger.info(f"Iteration $i - DNN Model - Test RMSE: $testError%1.4f")
    }
  
  val dnnWeights = dnnModels.last.weightMatrices
  logger.info(s"DNN weights:\n${
    dnnWeights.zipWithIndex.map{ case (m, i) => s"Layer $i:\n$m"}.mkString("\n")}"
  )
  
  val directSolveLFA = getLinearModel.solve(trainDataStream.head)
  val directSolveRMSE = directSolveLFA.rmse(testData)
  
  logger.info(f"Linear Model Direct Solve RMSE = $directSolveRMSE%1.6f\n-----------------------------")
  logger.info(f"Linear Model SGD (version 1)\n----------------")
  val (_, _, linearModel1) = trainDataStream
    .take(trainIterations)
    .zipWithIndex
    .foldLeft((0.0, 0.0, getLinearModel)) {
      case ((_, _, oldModel), (trainDataSeq, i)) =>
        val newModel = oldModel.update(trainDataSeq)
        val newTrainError = newModel.rmse(trainDataSeq)
        val newTestError = newModel.rmse(testData)
        logger.info(f"Iteration ${i + 1} - version 1 - Test RMSE: $newTestError%1.4f\tTrain RMSE: $newTrainError%1.4f")
        (newTrainError, newTestError, newModel)
    }
  
  logger.info(f"Linear Model SGD (version 2)\n----------------")
  val linearModels2 = getLinearModel
    .iterateUpdates(trainDataStream)
    .take(trainIterations + 1)
  
  linearModels2
    .zipWithIndex
    .foreach {
      case (f, i) =>
        val testError = f.rmse(testData)
        logger.info(f"Iteration $i - version 2 - Test RMSE: $testError%1.4f")
    }
  
  val linearModel2 = linearModels2.last
  
  val weightsDirectSolve = directSolveLFA.weights.weights
  val weightsSGDv1 = linearModel1.weights.weights
  val weightsSGDv2 = linearModel2.weights.weights
  val diff1 = linearModel1.distanceTo(directSolveLFA)
  
  logger.info(f"Linear Model Direct Solve Weights:\n$weightsDirectSolve\n---------------")
  logger.info(f"Linear Model SGD v1 Weights:\n$weightsSGDv1\n---------------")
  logger.info(f"Linear Model SGD v2 Weights:\n$weightsSGDv2\n---------------")
  logger.info(s"Distance between Direct Solve and SGD: $diff1")
  
}
