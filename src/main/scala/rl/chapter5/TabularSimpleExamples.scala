package rl.chapter5

import java.util.Locale

import rl.Tabular
import breeze.numerics._
import breeze.linalg._
import com.typesafe.scalalogging.Logger
import rl.utils.Gaussian
import TabularSimpleExamples._

object TabularSimpleExamples {
  
  type Triple = (Double, Double, Double)
  type AugmentedTriple = (Double, Double, Double, Double)
  type DataSeq = Seq[(Triple, Double)]
  
  val noiseDistribution: Gaussian = Gaussian(sigma = 2.0)
  
  def exampleModelDataGenerator: LazyList[DataSeq] = {
    
    val coefficients: AugmentedTriple = (2.0, 10.0, 4.0, -6.0)
    val (c0, c1, c2, c3) = coefficients
    val values = linspace(-10.0, 10.0, 21).valuesIterator.toSeq
    
    val points: Seq[(Double, Double, Double)] = for {
      x <- values
      y <- values
      z <- values
    } yield (x, y, z)
    
    def iterate: LazyList[DataSeq] = {
      val xy: DataSeq = points.map { point =>
        val (p1, p2, p3) = point
        val y = c0 + c1 * p1 + c2 * p2 + c3 * p3 + noiseDistribution.sample
        point -> y
      }
      xy #:: iterate
    }
    
    iterate
  }
  
}

object TabularSimpleExamplesApp extends App {
  val logger: Logger = Logger("TabularSimpleExamplesApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val trainIterations: Int = 30
  val trainDataGenerator: LazyList[DataSeq] = exampleModelDataGenerator
  val testData: DataSeq = exampleModelDataGenerator.head
  
  val tabular: Tabular[Triple] = Tabular[Triple]()
  
  logger.info(f"Tabular Model (version 1)\n----------------")
  trainDataGenerator
    .take(trainIterations)
    .zipWithIndex
    .foldLeft((0.0, 0.0, tabular)) {
    case ((_, _, oldModel), (trainDataSeq, i)) =>
      val newModel = oldModel.update(trainDataSeq)
      val newTrainError = newModel.rmse(trainDataSeq)
      val newTestError = newModel.rmse(testData)
      logger.info(f"Iteration $i - Test RMSE: $newTestError%1.4f\tTrain RMSE: $newTrainError%1.4f")
      (newTrainError, newTestError, newModel)
  }
  
  logger.info(f"Tabular Model (version 2)\n----------------")
  tabular
    .iterateUpdates(trainDataGenerator)
    .take(trainIterations + 1)
    .zipWithIndex
    .foreach {
      case (f, i) =>
        val testError = f.rmse(testData)
        logger.info(f"Iteration $i - Test RMSE: $testError%1.4f")
    }
  
}
