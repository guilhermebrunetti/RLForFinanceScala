package rl

import java.util.Locale

import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import com.typesafe.scalalogging.Logger
import rl.AdamGradientDefaults._
import rl.DNNApproximation.getInputOutputDimensions
import rl.Tabular.defaultCountToWeight
import rl.Weights.EPSILON
import rl.utils.MapAlgebra._
import rl.utils.{Gaussian, IterableVectorSpace, VectorSpace, VectorSpaceOperators}

/**
 * An interface for different kinds of function approximations
 * (tabular, linear, DNN... etc), with several implementations.
 */

case class Gradient[X, +F[_] <: FunctionApproximation[_]](
  functionApproximation: F[X]
)

object FunctionApproximationUtils {
  
  def add[X, F[_] <: FunctionApproximation[_]](x: Gradient[X, F], y: Gradient[X, F]): Gradient[X, F] = {
    Gradient(add(x.functionApproximation, y.functionApproximation))
  }
  
  def add[X, F[_] <: FunctionApproximation[_]](x: F[X], y: F[X]): F[X] = {
    val res = (x, y) match {
      case (a: Dynamic[X]@unchecked, b: Dynamic[X]@unchecked) => a + b
      case (a: Tabular[X]@unchecked, b: Tabular[X]@unchecked) => a + b
      case (a: LinearFunctionApproximation[X]@unchecked, b: LinearFunctionApproximation[X]@unchecked) => a + b
      case (a: DNNApproximation[X]@unchecked, b: DNNApproximation[X]@unchecked) => a + b
      case _ => throw new IllegalArgumentException(s"Incompatible classes between x (${x.getClass}) and y (${y.getClass})")
    }
    
    res.asInstanceOf[F[X]]
  }
  
  def multiply[X, F[_] <: FunctionApproximation[_]](x: Gradient[X, F], scalar: Double): Gradient[X, F] = {
    Gradient(multiply(x.functionApproximation, scalar))
  }
  
  def multiply[X, F[_] <: FunctionApproximation[_]](x: F[X], scalar: Double): F[X] = {
    val res = x match {
      case a: Dynamic[X]@unchecked => a * scalar
      case a: Tabular[X]@unchecked => a * scalar
      case a: LinearFunctionApproximation[X]@unchecked => a * scalar
      case a: DNNApproximation[X]@unchecked => a * scalar
      case _ => throw new IllegalArgumentException(s"Unsupported class (${x.getClass})")
    }
    
    res.asInstanceOf[F[X]]
  }
  
  def objectiveGradient[X, F[_] <: FunctionApproximation[_]](
    f: F[X],
    xySeq: Iterable[(X, Double)],
    derivativeFunction: (X, Double) => Double
  ): Gradient[X, F] = {
    
    val res = f match {
      case a: Dynamic[X]@unchecked => a.objectiveGradient(xySeq, derivativeFunction)
      case a: Tabular[X]@unchecked => a.objectiveGradient(xySeq, derivativeFunction)
      case a: LinearFunctionApproximation[X]@unchecked => a.objectiveGradient(xySeq, derivativeFunction)
      case a: DNNApproximation[X]@unchecked => a.objectiveGradient(xySeq, derivativeFunction)
      case _ => throw new IllegalArgumentException(s"Unsupported class (${f.getClass})")
    }
    
    res.asInstanceOf[Gradient[X, F]]
  }
  
  def updateWithGradient[X, F[_] <: FunctionApproximation[_]](
    f: F[X],
    gradient: Gradient[X, F]
  ): F[X] = {
  
    val res = (f, gradient) match {
      case (a: Dynamic[X]@unchecked, g: Gradient[X, Dynamic]@unchecked) =>
        a.updateWithGradient(g)
      case (a: Tabular[X]@unchecked, g: Gradient[X, Tabular]@unchecked) =>
        a.updateWithGradient(g)
      case (a: LinearFunctionApproximation[X]@unchecked, g: Gradient[X, LinearFunctionApproximation]@unchecked) =>
        a.updateWithGradient(g)
      case (a: DNNApproximation[X]@unchecked, g: Gradient[X, DNNApproximation]@unchecked) =>
        a.updateWithGradient(g)
      case _ => throw new IllegalArgumentException(s"Unsupported class (${f.getClass})")
    }
  
    res.asInstanceOf[F[X]]
  }
  
  def updateEligibilityTrace[X, F[_] <: FunctionApproximation[_]](
    f: F[X],
    eligibilityTrace: Gradient[X, F],
    weight: Double, // gamma, gamma * lambda, etc
    xySeq: Iterable[(X, Double)]
  ): Gradient[X, F] = {
    
    val res = (f, eligibilityTrace) match {
      case (a: Dynamic[X]@unchecked, g: Gradient[X, Dynamic]@unchecked) =>
        a.updateEligibilityTrace(g, weight, xySeq)
      case (a: Tabular[X]@unchecked, g: Gradient[X, Tabular]@unchecked) =>
        a.updateEligibilityTrace(g, weight, xySeq)
      case (a: LinearFunctionApproximation[X]@unchecked, g: Gradient[X, LinearFunctionApproximation]@unchecked) =>
        a.updateEligibilityTrace(g, weight, xySeq)
      case (a: DNNApproximation[X]@unchecked, g: Gradient[X, DNNApproximation]@unchecked) =>
        a.updateEligibilityTrace(g, weight, xySeq)
      case _ => throw new IllegalArgumentException(s"Unsupported class (${f.getClass})")
    }
    
    res.asInstanceOf[Gradient[X, F]]
  }
  
  def gradientZero[X, F[_] <: FunctionApproximation[_]](f: F[X]): Gradient[X, F] = {
    val res = f match {
      case a: Dynamic[X]@unchecked => a.gradientZero
      case a: Tabular[X]@unchecked => a.gradientZero
      case a: LinearFunctionApproximation[X]@unchecked => a.gradientZero
      case a: DNNApproximation[X]@unchecked => a.gradientZero
      case _ => throw new IllegalArgumentException(s"Unsupported class (${f.getClass})")
    }
    
    res.asInstanceOf[Gradient[X, F]]
  }
  
  def updateWithEligibilityTrace[X, F[_] <: FunctionApproximation[_]](
    f: F[X],
    eligibilityTrace: Gradient[X, F],
    x: X,
    y: Double
  ): F[X] = {
    
    val res = (f, eligibilityTrace) match {
      case (a: Dynamic[X]@unchecked, g: Gradient[X, Dynamic]@unchecked) =>
        a.updateWithEligibilityTrace(g, x, y)
      case (a: Tabular[X]@unchecked, g: Gradient[X, Tabular]@unchecked) =>
        a.updateWithEligibilityTrace(g, x, y)
      case (a: LinearFunctionApproximation[X]@unchecked, g: Gradient[X, LinearFunctionApproximation]@unchecked) =>
        a.updateWithEligibilityTrace(g, x, y)
      case (a: DNNApproximation[X]@unchecked, g: Gradient[X, DNNApproximation]@unchecked) =>
        a.updateWithEligibilityTrace(g, x, y)
      case _ => throw new IllegalArgumentException(s"Unsupported class (${f.getClass})")
    }
    
    res.asInstanceOf[F[X]]
  }
  
}

trait FunctionApproximationHelper[X, F[_] <: FunctionApproximation[_]]
  extends VectorSpaceOperators[F[X]] {
  self: F[X] =>
  
  /**
   * Update the internal parameters of self FunctionApprox using the
   * input gradient that is presented as a Gradient[FunctionApprox]
   */
  def updateWithGradient(gradient: Gradient[X, F]): F[X]
  
  /**
   * Computes the gradient of an objective function of the self
   * FunctionApprox with respect to the parameters in the internal
   * representation of the FunctionApprox. The gradient is output
   * in the form of a Gradient[FunctionApprox] whose internal parameters are
   * equal to the gradient values. The argument 'derivativeFunction'
   * represents the derivative of the objective with respect to the output
   * (evaluate) of the FunctionApprox, when evaluated at a Sequence of
   * x values and a Sequence of y values (to be obtained from 'xySeq')
   */
  def objectiveGradient(
    xySeq: Iterable[(X, Double)],
    derivativeFunction: (X, Double) => Double
  ): Gradient[X, F]
  
  /**
   * Update the internal parameters of the FunctionApprox
   * based on incremental data provided in the form of (x,y)
   * pairs as a xySeq data structure
   */
  def update(xySeq: Iterable[(X, Double)]): F[X] = {
    def derivativeFunction(x: X, y: Double): Double = {
      self.asInstanceOf[FunctionApproximation[X]].apply(x) - y
    }
    
    updateWithGradient(objectiveGradient(xySeq, derivativeFunction))
  }
  
  def gradientZero: Gradient[X, F]
  
  def updateEligibilityTrace(
    eligibilityTrace: Gradient[X, F],
    weight: Double, // gamma, gamma * lambda, etc
    xySeq: Iterable[(X, Double)]
  ): Gradient[X, F]
  
  def updateWithEligibilityTrace(
    eligibilityTrace: Gradient[X, F],
    x: X,
    y: Double
  ): F[X]
  
}

/**
 * Interface for function approximations.
 * An object of this class approximates some function X -> Double in a way
 * that can be evaluated at specific points in X and updated with
 * additional (X, Double) points.
 */
trait FunctionApproximation[X] {
  
  def apply(x: X): Double
  
  /**
   * Update the internal parameters of the FunctionApprox
   * based on incremental data provided in the form of (x,y)
   * pairs as a xySeq data structure
   */
  def update(xySeq: Iterable[(X, Double)]): FunctionApproximation[X]
  
  /**
   * Assuming the entire data set of (x,y) pairs is available
   * in the form of the given input xySeq data structure,
   * solve for the internal parameters of the FunctionApprox
   * such that the internal parameters are fitted to xySeq.
   * Since this is a best-fit, the internal parameters are fitted
   * to within the input errorTolerance (where applicable, since
   * some methods involve a direct solve for the fit that don't
   * require an errorTolerance)
   */
  def solve(
    xySeq: Iterable[(X, Double)],
    errorTolerance: Option[Double] = None
  ): FunctionApproximation[X]
  
  /**
   * Given a stream (Iterable) of data sets of (x,y) pairs,
   * perform a series of incremental updates to the internal
   * parameters (using update method), with each internal
   * parameter update done for each data set of (x,y) pairs in the
   * input stream of xySeqStream
   */
  def iterateUpdates(
    xySeqStream: Iterable[Iterable[(X, Double)]]
  ): Iterable[FunctionApproximation[X]]
  
  /**
   * Given a stream (Iterator) of data sets of (x,y) pairs,
   * perform a series of incremental updates to the internal
   * parameters (using update method), with each internal
   * parameter update done for each data set of (x,y) pairs in the
   * input stream of xySeqStream
   */
  def iterateUpdates(
    xySeqStream: Iterator[Iterable[(X, Double)]]
  ): Iterator[FunctionApproximation[X]]
  
  /**
   * The Root-Mean-Squared-Error between FunctionApproximation's
   * predictions (from evaluate) and the associated (supervisory)
   * y values
   */
  def rmse(xySeq: Iterable[(X, Double)]): Double = {
    val (xSeq, ySeq) = xySeq.unzip
    val errors: DenseVector[Double] = this.evaluate(xSeq) - DenseVector(ySeq.toArray)
    val mse = mean(errors.map(x => x * x))
    sqrt(mse)
  }
  
  /**
   * Computes expected value of y for each x in
   * xSeq (with the probability distribution
   * function of y|x estimated as FunctionApprox)
   */
  def evaluate(xSeq: Iterable[X]): DenseVector[Double] = {
    val ys = xSeq.map(this.apply)
    DenseVector(ys.toArray)
  }
  
  /**
   * Return the input X that maximizes the function being approximated.
   * Arguments:
   * xSeq -- list of inputs to evaluate and maximize, cannot be empty
   * Returns the X that maximizes the function this approximates.
   */
  def argMax(xSeq: Iterable[X]): X = {
    val args = xSeq.toSeq
    args.maxBy(apply)
  }
  
  def zero: FunctionApproximation[X]
  
}

/**
 * A FunctionApprox that works exactly the same as exact dynamic
 * programming. Each update for a value in X replaces the previous
 * value at X altogether.
 *
 * Fields:
 * valuesMap -- mapping from X to its approximated value
 */
class Dynamic[X](val valuesMap: Map[X, Double])
  extends FunctionApproximation[X] with FunctionApproximationHelper[X, Dynamic] {
  
  override val op: VectorSpace[Dynamic[X]] = Dynamic.vectorSpace[X]
  
  override def solve(
    xySeq: Iterable[(X, Double)],
    errorTolerance: Option[Double]): Dynamic[X] = {
    Dynamic(xySeq.toMap)
  }
  
  // Implementing the method in each class to overcome issue with output type
  override def iterateUpdates(
    xySeqStream: Iterable[Iterable[(X, Double)]]
  ): Iterable[Dynamic[X]] = {
    IterateUtils.accumulate(
      xySeqStream,
      this)(
      { case (fa, xy) => fa.update(xy) }
    )
  }
  
  override def gradientZero: Gradient[X, Dynamic] = Gradient(zero)
  
  override def zero: Dynamic[X] = this * 0.0
  
  override def updateEligibilityTrace(
    eligibilityTrace: Gradient[X, Dynamic],
    weight: Double,
    xySeq: Iterable[(X, Double)]
  ): Gradient[X, Dynamic] = {
    val f = eligibilityTrace.functionApproximation
    val g = this.objectiveGradient(xySeq, (_, _) => 1.0).functionApproximation
    Gradient((f * weight) + g)
  }
  
  def objectiveGradient(
    xySeq: Iterable[(X, Double)],
    derivativeFunction: (X, Double) => Double
  ): Gradient[X, Dynamic] = {
    val derivatives: Map[X, Double] = xySeq.map { case (x, y) => x -> derivativeFunction(x, y) }.toMap
    Gradient(Dynamic(derivatives))
  }
  
  override def updateWithEligibilityTrace(
    eligibilityTrace: Gradient[X, Dynamic],
    x: X,
    y: Double
  ): Dynamic[X] = {
    val f = eligibilityTrace.functionApproximation
    val error = this.apply(x) - y
    updateWithGradient(Gradient(f * error))
  }
  
  override def apply(x: X): Double = valuesMap.getOrElse(x, 0)
  
  def updateWithGradient(gradient: Gradient[X, Dynamic]): Dynamic[X] = {
    val newMap: Dynamic[X] = this - gradient.functionApproximation
    newMap
  }
  
  override def iterateUpdates(
    xySeqStream: Iterator[Iterable[(X, Double)]]
  ): Iterator[Dynamic[X]] = {
    IterateUtils.accumulate(
      xySeqStream,
      this)(
      { case (fa, xy) => fa.update(xy) }
    )
  }
  
}

object Dynamic {
  
  def vectorSpace[X]: VectorSpace[Dynamic[X]] =
    new VectorSpace[Dynamic[X]] {
      override def add(x: Dynamic[X], y: Dynamic[X]): Dynamic[X] = {
        Dynamic(sumMap(x.valuesMap, y.valuesMap))
      }
      
      override def multiply(x: Dynamic[X], scalar: Double): Dynamic[X] = {
        Dynamic(multiplyByScalar(x.valuesMap, scalar))
      }
      
      override def norm(x: Dynamic[X]): Double = x.valuesMap.values.map(abs(_)).max
    }
  
  def apply[X](valuesMap: Map[X, Double] = Map.empty[X, Double]): Dynamic[X] = new Dynamic(valuesMap)
  
}

/**
 * Approximates a function with a discrete domain (X), without any
 * interpolation. The value for each X is maintained as a weighted
 * mean of observations by recency (managed by countToWeight function).
 *
 * In practice, this means you can use this to approximate a function
 * with a learning rate alpha(n) specified by countToWeight.
 *
 * If countToWeightFunc always returns 1, this behaves the same
 * way as Dynamic.
 *
 * Fields:
 * valuesMap -- mapping from X to its approximated value
 * countsMap -- how many times a given X has been updated
 * countToWeight -- function for how much to weigh an update
 * to X based on the number of times that X has been updated
 */
class Tabular[X](
  val valuesMap: Map[X, Double],
  val countsMap: Map[X, Int],
  val countToWeight: Int => Double
) extends FunctionApproximation[X] with FunctionApproximationHelper[X, Tabular] {
  require(valuesMap.keySet == countsMap.keySet,
    s"valuesMap and countsMap should have same keySet. Instead, got:\nvaluesMap.keySet = ${valuesMap.keySet}\ncountsMap.keySet = ${countsMap.keySet}")
  
  override def op: VectorSpace[Tabular[X]] = Tabular.vectorSpace[X]
  
  def countsMapToString: String = {
    val total = countsMap.values.sum
    val strings: Seq[String] =
      countsMap.map { case (x, value) => f"Count for $x: $value%2d" }.toSeq :+ f"Total counts: $total%2d"
    strings.mkString("\n")
  }
  
  override def solve(xySeq: Iterable[(X, Double)], errorTolerance: Option[Double]): Tabular[X] = {
    val initialPoint = (Map.empty[X, Double], Map.empty[X, Int])
    
    val (values: Map[X, Double], counts: Map[X, Int]) =
      xySeq.foldLeft(initialPoint) { case ((oldValues, oldCounts), (x, y)) =>
        val newCounts = oldCounts.updated(x, oldCounts.getOrElse(x, 0) + 1)
        val weight = this.countToWeight(newCounts(x))
        val newValues = oldValues.updated(x, weight * y + (1 - weight) * oldValues.getOrElse(x, 0.0))
        (newValues, newCounts)
      }
    
    Tabular(values, counts, this.countToWeight)
  }
  
  // Implementing the method in each class to overcome issue with output type
  override def iterateUpdates(
    xySeqStream: Iterable[Iterable[(X, Double)]]
  ): Iterable[Tabular[X]] = {
    IterateUtils.accumulate(
      xySeqStream,
      this)(
      { case (fa, xy) => fa.update(xy) }
    )
  }
  
  override def iterateUpdates(
    xySeqStream: Iterator[Iterable[(X, Double)]]
  ): Iterator[Tabular[X]] = {
    IterateUtils.accumulate(
      xySeqStream,
      this)(
      { case (fa, xy) => fa.update(xy) }
    )
  }
  
  override def gradientZero: Gradient[X, Tabular] = Gradient(zero)
  
  override def zero: Tabular[X] = copy(
    valuesMap = this.valuesMap.map { case (k, _) => k -> 0.0 },
    countsMap = this.countsMap.map { case (k, _) => k -> 0 },
  )
  
  def copy(
    valuesMap: Map[X, Double] = this.valuesMap,
    countsMap: Map[X, Int] = this.countsMap,
    countToWeight: Int => Double = this.countToWeight
  ): Tabular[X] = Tabular(valuesMap, countsMap, countToWeight)
  
  override def updateEligibilityTrace(
    eligibilityTrace: Gradient[X, Tabular],
    weight: Double,
    xySeq: Iterable[(X, Double)]
  ): Gradient[X, Tabular] = {
    val f = eligibilityTrace.functionApproximation.resetCount
    val g = this.objectiveGradient(xySeq, (_, _) => 1.0).functionApproximation
    val h = f * weight
    val l = h + g
    Gradient(l)
  }
  
  override def objectiveGradient(
    xySeq: Iterable[(X, Double)],
    derivativeFunction: (X, Double) => Double
  ): Gradient[X, Tabular] = {
  
    val initialPoint = (Map.empty[X, Double], Map.empty[X, Int])
    val (derivatives: Map[X, Double], counts: Map[X, Int]) =
      xySeq.foldLeft(initialPoint) { case ((oldValues, oldCounts), (x, y)) =>
        val newCounts = oldCounts.updated(x, oldCounts.getOrElse(x, 0) + 1)
        val weight = defaultCountToWeight(newCounts(x))
        val derivative = derivativeFunction(x, y)
        val newValues = oldValues.updated(x, weight * derivative + (1 - weight) * oldValues.getOrElse(x, 0.0))
        (newValues, newCounts)
      }
    
    Gradient(Tabular(derivatives, counts, this.countToWeight))
  }
  
  protected def resetCount: Tabular[X] = copy(
    countsMap = this.countsMap.map { case (k, _) => k -> 0 }
  )
  
  override def updateWithEligibilityTrace(eligibilityTrace: Gradient[X, Tabular], x: X, y: Double): Tabular[X] = {
    val f = eligibilityTrace.functionApproximation
    val error = this.apply(x) - y
    updateWithGradient(Gradient(f * error))
  }
  
  override def apply(x: X): Double = valuesMap.getOrElse(x, 0)
  
  /**
   * Update the approximation with the given gradient.
   * Each X keeps a count n of how many times it was updated, and
   * each subsequent update is scaled by countToWeight(n),
   * which defines our learning rate.
   */
  override def updateWithGradient(gradient: Gradient[X, Tabular]): Tabular[X] = {
    val f: Tabular[X] = gradient.functionApproximation
    
    val (newValues, newCounts) = (f.valuesMap.keySet ++ this.valuesMap.keySet).map { k =>
      val newCount = f.countsMap.getOrElse(k, 0) + this.countsMap.getOrElse(k, 0)
      val weight = countToWeight(newCount)
      val currentValue = this.valuesMap.getOrElse(k, 0.0)
      val shift = weight * f.valuesMap.getOrElse(k, 0.0)
      val newValue = currentValue - shift
      (k -> newValue, k -> newCount)
    }.unzip
    
    //    val newCounts = sumMap(this.countsMap, f.countsMap)
    //    val weights = newCounts.view.mapValues(countToWeight).toMap
    //    val weightedValuesMap = productMap(f.valuesMap, weights)
    //    val newValues = subtractMap(this.valuesMap, weightedValuesMap)
    
    this.copy(newValues.toMap, newCounts.toMap)
  }
}

object Tabular {
  
  def defaultCountToWeight(n: Int): Double = if (n > 0) 1.0 / n else 0.0
  
  def vectorSpace[X]: VectorSpace[Tabular[X]] =
    new VectorSpace[Tabular[X]] {
      override def add(x: Tabular[X], y: Tabular[X]): Tabular[X] = {
        Tabular(
          sumMap(x.valuesMap, y.valuesMap),
          sumMap(x.countsMap, y.countsMap)
        )
      }
      
      override def multiply(x: Tabular[X], scalar: Double): Tabular[X] = {
        Tabular(multiplyByScalar(x.valuesMap, scalar), x.countsMap)
      }
      
      override def norm(x: Tabular[X]): Double = x.valuesMap.values.map(abs(_)).max
    }
  
  def apply[X](
    valuesMap: Map[X, Double] = Map.empty[X, Double],
    countsMap: Map[X, Int] = Map.empty[X, Int],
    countToWeight: Int => Double = defaultCountToWeight
  ): Tabular[X] = {
    val countsMapOverride: Map[X, Int] = if (valuesMap.isEmpty)
      countsMap
    else {
      if (countsMap.isEmpty) valuesMap.view.mapValues(_ => 0).toMap else countsMap
    }
    new Tabular(valuesMap, countsMapOverride, countToWeight)
  }
  
  def learningRateSchedule(
    initialLearningRate: Double,
    halfLife: Double,
    exponent: Double
  )(n: Int): Double = {
    if (n > 0) {
      initialLearningRate * pow(1 + (n - 1) / halfLife, -exponent)
    }
    else 0.0
  }
}

object AdamGradientDefaults {
  val defaultLearningRate: Double = 0.001
  val defaultDecay1: Double = 0.9
  val defaultDecay2: Double = 0.999
  
  def defaultSettings: AdamGradient = AdamGradient()
}

case class AdamGradient(
  learningRate: Double = defaultLearningRate,
  decay1: Double = defaultDecay1,
  decay2: Double = defaultDecay2
)

class Weights(
  val adamGradient: AdamGradient,
  val time: Int,
  val weights: DenseVector[Double],
  val adamCache1: DenseVector[Double],
  val adamCache2: DenseVector[Double]) extends VectorSpaceOperators[Weights] {
  require(
    Set(weights, adamCache1, adamCache2).map(_.length).size == 1,
    s"All input vectors should have same size. Instead, got ${Set(weights, adamCache1, adamCache2).map(_.length).size}"
  )
  
  def update(gradientVector: DenseVector[Double]): Weights = {
    require(
      gradientVector.length == weights.length,
      s"gradientVector needs to have same length as weights."
    )
    
    val newTime: Int = time + 1
    val learningRate = adamGradient.learningRate
    val decay1 = adamGradient.decay1
    val decay2 = adamGradient.decay2
    val newAdamCache1: DenseVector[Double] = {
      decay1 * adamCache1 + (1 - decay1) * gradientVector
    }
    val newAdamCache2: DenseVector[Double] = {
      decay2 * adamCache2 + (1 - decay2) * pow(gradientVector, 2)
    }
    
    val correctedM = newAdamCache1 / (1 - pow(decay1, newTime))
    val correctedV = newAdamCache2 / (1 - pow(decay2, newTime))
    
    val direction = correctedM / (sqrt(correctedV) + EPSILON)
    val newWeights = weights - learningRate * direction
    
    copy(
      time = newTime,
      weights = newWeights,
      adamCache1 = newAdamCache1,
      adamCache2 = newAdamCache2,
    )
  }
  
  def copy(
    adamGradient: AdamGradient = this.adamGradient,
    time: Int = this.time,
    weights: DenseVector[Double] = this.weights,
    adamCache1: DenseVector[Double] = this.adamCache1,
    adamCache2: DenseVector[Double] = this.adamCache2
  ): Weights = Weights(adamGradient, time, weights, adamCache1, adamCache2)
  
  override def op: VectorSpace[Weights] = Weights.vectorSpace
}

object Weights {
  
  val EPSILON: Double = 1.0e-6
  
  def vectorSpace: VectorSpace[Weights] =
    new VectorSpace[Weights] {
      override def add(x: Weights, y: Weights): Weights =
        x.copy(weights = x.weights + y.weights)
      
      override def multiply(x: Weights, scalar: Double): Weights =
        x.copy(weights = x.weights * scalar)
      
      override def norm(x: Weights): Double = x.weights.valuesIterator.map(x => abs(x)).max
    }
  
  def create(
    weights: DenseVector[Double],
    adamGradient: AdamGradient = defaultSettings,
    adamCache1Option: Option[DenseVector[Double]] = None,
    adamCache2Option: Option[DenseVector[Double]] = None
  ): Weights = {
    apply(adamGradient, time = 0, weights, adamCache1Option, adamCache2Option)
  }
  
  def apply(
    adamGradient: AdamGradient = defaultSettings,
    time: Int = 0,
    weights: DenseVector[Double],
    adamCache1Option: Option[DenseVector[Double]] = None,
    adamCache2Option: Option[DenseVector[Double]] = None
  ): Weights = {
    new Weights(
      adamGradient,
      time,
      weights,
      adamCache1Option.getOrElse(DenseVector.zeros[Double](weights.length)),
      adamCache2Option.getOrElse(DenseVector.zeros[Double](weights.length))
    )
  }
  
  def apply(
    adamGradient: AdamGradient,
    time: Int,
    weights: DenseVector[Double],
    adamCache1: DenseVector[Double],
    adamCache2: DenseVector[Double]
  ): Weights = {
    new Weights(adamGradient, time, weights, adamCache1, adamCache2)
  }
}

trait HasFeatureFunctions[X] {
  
  def featureFunctions: Seq[X => Double]
  
  def getFeatureValues(xSeq: Iterable[X]): DenseMatrix[Double] = {
    val features: Seq[Seq[Double]] = xSeq.toSeq.map { x =>
      featureFunctions.map(f => f(x))
    }
    DenseMatrix.apply(features: _*)
  }
}

class LinearFunctionApproximation[X](
  val featureFunctions: Seq[X => Double],
  val regularizationCoefficient: Double,
  val weights: Weights,
  val directSolve: Boolean
) extends FunctionApproximation[X]
  with FunctionApproximationHelper[X, LinearFunctionApproximation]
  with HasFeatureFunctions[X] {
  require(
    featureFunctions.size == weights.weights.length,
    s"featureFunctions and weights should have same length. Instead, got featureFunctions.size: ${featureFunctions.size} and weights.weights.length: ${weights.weights.length} "
  )
  
  /**
   * solve the optimal weights either analytically or using iterations
   */
  override def solve(
    xySeq: Iterable[(X, Double)],
    errorTolerance: Option[Double]
  ): LinearFunctionApproximation[X] = if (directSolve) {
    val (xVals, yVals) = xySeq.unzip
    val featureValues = getFeatureValues(xVals)
    val yVector = DenseVector(yVals.toArray)
    val hMatrix = featureValues.t * featureValues
    val const = xVals.size * regularizationCoefficient
    val n = featureFunctions.size
    val regMatrix = const * DenseMatrix.eye[Double](n)
    val A = hMatrix + regMatrix
    val b = featureValues.t * yVector
    val newWeights = A \ b // solution of linear system Ax = b
    this.copy(weights = this.weights.copy(weights = newWeights))
  }
  else {
    val tolerance = errorTolerance.getOrElse(EPSILON)
    
    def done(
      a: LinearFunctionApproximation[X],
      b: LinearFunctionApproximation[X],
    ): Boolean = a.within(b, tolerance)
    
    IterateUtils.converged(
      iterateUpdates(Iterator.continually(xySeq)),
      done)
  }
  
  // Implementing the method in each class to overcome issue with output type
  override def iterateUpdates(
    xySeqStream: Iterable[Iterable[(X, Double)]]
  ): Iterable[LinearFunctionApproximation[X]] = {
    IterateUtils.accumulate(
      xySeqStream,
      this)(
      { case (fa, xy) => fa.update(xy) }
    )
  }
  
  override def op: VectorSpace[LinearFunctionApproximation[X]] =
    LinearFunctionApproximation.vectorSpace[X]
  
  override def gradientZero: Gradient[X, LinearFunctionApproximation] = Gradient(zero)
  
  override def zero: LinearFunctionApproximation[X] = this * 0.0
  
  override def updateEligibilityTrace(
    eligibilityTrace: Gradient[X, LinearFunctionApproximation],
    weight: Double,
    xySeq: Iterable[(X, Double)]
  ): Gradient[X, LinearFunctionApproximation] = {
    val f = eligibilityTrace.functionApproximation
    val g = this.objectiveGradient(xySeq, (_, _) => 1.0).functionApproximation
    Gradient((f * weight) + g)
  }
  
  override def objectiveGradient(
    xySeq: Iterable[(X, Double)],
    derivativeFunction: (X, Double) => Double
  ): Gradient[X, LinearFunctionApproximation] = {
    val xSeq = xySeq.map(_._1)
    val derivatives = xySeq.map { case (x, y) => derivativeFunction(x, y) }
    val derivativeVector = DenseVector(derivatives.toArray)
    val n = derivativeVector.length.toDouble
    val featureValues = getFeatureValues(xSeq)
    val regularizationVector = regularizationCoefficient * weights.weights
    val gradient: DenseVector[Double] = featureValues.t * derivativeVector / n + regularizationVector
    Gradient(this.copy(weights = weights.copy(weights = gradient)))
  }
  
  override def updateWithEligibilityTrace(
    eligibilityTrace: Gradient[X, LinearFunctionApproximation],
    x: X,
    y: Double
  ): LinearFunctionApproximation[X] = {
    val f = eligibilityTrace.functionApproximation
    val error = this.apply(x) - y
    updateWithGradient(Gradient(f * error))
  }
  
  override def apply(x: X): Double = featureFunctions
    .zip(weights.weights.valuesIterator)
    .map { case (f, w) => w * f(x) }
    .sum
  
  override def updateWithGradient(
    gradient: Gradient[X, LinearFunctionApproximation]
  ): LinearFunctionApproximation[X] = {
    val gradientVector = gradient.functionApproximation.weights.weights
    this.copy(weights = this.weights.update(gradientVector))
  }
  
  def copy(
    featureFunctions: Seq[X => Double] = this.featureFunctions,
    regularizationCoefficient: Double = this.regularizationCoefficient,
    weights: Weights = this.weights,
    directSolve: Boolean = this.directSolve
  ): LinearFunctionApproximation[X] =
    LinearFunctionApproximation(featureFunctions, regularizationCoefficient, weights, directSolve)
  
  override def iterateUpdates(
    xySeqStream: Iterator[Iterable[(X, Double)]]
  ): Iterator[LinearFunctionApproximation[X]] = {
    IterateUtils.accumulate(
      xySeqStream,
      this)(
      { case (fa, xy) => fa.update(xy) }
    )
  }
}

object LinearFunctionApproximation {
  
  def create[X](
    featureFunctions: Seq[X => Double],
    adamGradient: AdamGradient = defaultSettings,
    regularizationCoefficient: Double = 0.0,
    weightsOption: Option[Weights] = None,
    directSolve: Boolean = true
  ): LinearFunctionApproximation[X] = {
    apply(
      featureFunctions,
      regularizationCoefficient,
      weightsOption.getOrElse(
        Weights.create(
          DenseVector.zeros[Double](featureFunctions.size),
          adamGradient)
      ),
      directSolve
    )
  }
  
  def apply[X](
    featureFunctions: Seq[X => Double],
    regularizationCoefficient: Double = 0.0,
    weights: Weights,
    directSolve: Boolean = true
  ): LinearFunctionApproximation[X] =
    new LinearFunctionApproximation(featureFunctions, regularizationCoefficient, weights, directSolve)
  
  def vectorSpace[X]: VectorSpace[LinearFunctionApproximation[X]] =
    new VectorSpace[LinearFunctionApproximation[X]] {
      override def add(
        x: LinearFunctionApproximation[X],
        y: LinearFunctionApproximation[X]
      ): LinearFunctionApproximation[X] = x.copy(weights = x.weights + y.weights)
      
      override def multiply(
        x: LinearFunctionApproximation[X],
        scalar: Double
      ): LinearFunctionApproximation[X] = x.copy(weights = x.weights * scalar)
      
      override def norm(x: LinearFunctionApproximation[X]): Double = x.weights.norm
    }
  
}

case class DNNSpec(
  neurons: Seq[Int],
  bias: Boolean,
  hiddenActivation: Double => Double,
  hiddenActivationDerivative: Double => Double,
  outputActivation: Double => Double,
  outputActivationDerivative: Double => Double
)

class DNNApproximation[X](
  val featureFunctions: Seq[X => Double],
  val dnnSpec: DNNSpec,
  val regularizationCoefficient: Double,
  val weights: Seq[Weights],
) extends FunctionApproximation[X]
  with FunctionApproximationHelper[X, DNNApproximation]
  with HasFeatureFunctions[X] {
  
  lazy val weightMatrices: Seq[DenseMatrix[Double]] =
    weights
      .zip(inputOutputDimensions)
      .map { case (w, (dimIn, dimOut)) =>
        vectorToMatrix(w.weights, dimIn, dimOut)
      }
  
  val inputOutputDimensions: Seq[(Int, Int)] = getInputOutputDimensions(featureFunctions, dnnSpec)
  
  override def evaluate(xSeq: Iterable[X]): DenseVector[Double] = forwardPropagation(xSeq).last.toDenseVector
  
  /**
   *
   * @param xSeq: Iterable of X
   * @return list of length (L+2) where the first (L+1) values
   *         each represent the 2-D input arrays (of size n x |I_l|),
   *         for each of the (L+1) layers (L of which are hidden layers),
   *         and the last value represents the output of the DNN (as a
   *         1-D array of length n)
   */
  def forwardPropagation(
    xSeq: Iterable[X]
  ): Seq[DenseMatrix[Double]] = {
    val inputs: DenseMatrix[Double] = getFeatureValues(xSeq)
    val weights = weightMatrices
    val res: Seq[DenseMatrix[Double]] = weights
      .init
      .scanLeft(inputs) { case (input, weightMatrix) =>
        val output: DenseMatrix[Double] = (input * weightMatrix).map(dnnSpec.hiddenActivation)
        if (dnnSpec.bias) {
          DenseMatrix.horzcat(DenseMatrix.ones[Double](output.rows, 1), output)
        } else
          output
      }
    res.appended {
      val input: DenseMatrix[Double] = res.last
      val weightMatrix: DenseMatrix[Double] = weights.last
      (input * weightMatrix).map(dnnSpec.outputActivation)
    }
  }
  
  override def solve(
    xySeq: Iterable[(X, Double)],
    errorTolerance: Option[Double]
  ): DNNApproximation[X] = {
    val tolerance = errorTolerance.getOrElse(EPSILON)
    
    def done(
      a: DNNApproximation[X],
      b: DNNApproximation[X],
    ): Boolean = a.within(b, tolerance)
    
    IterateUtils.converged(
      iterateUpdates(Iterator.continually(xySeq)),
      done)
  }
  
  // Implementing the method in each class to overcome issue with output type
  override def iterateUpdates(
    xySeqStream: Iterable[Iterable[(X, Double)]]
  ): Iterable[DNNApproximation[X]] = {
    IterateUtils.accumulate(
      xySeqStream,
      this)(
      { case (fa, xy) => fa.update(xy) }
    )
  }
  
  override def op: VectorSpace[DNNApproximation[X]] = DNNApproximation.vectorSpace[X]
  
  override def gradientZero: Gradient[X, DNNApproximation] = Gradient(zero)
  
  override def zero: DNNApproximation[X] = this * 0.0
  
  override def updateEligibilityTrace(
    eligibilityTrace: Gradient[X, DNNApproximation],
    weight: Double,
    xySeq: Iterable[(X, Double)]
  ): Gradient[X, DNNApproximation] = {
    val f = eligibilityTrace.functionApproximation
    val g = this.objectiveGradient(xySeq, (_, _) => 1.0).functionApproximation
    Gradient((f * weight) + g)
  }
  
  override def objectiveGradient(
    xySeq: Iterable[(X, Double)],
    derivativeFunction: (X, Double) => Double
  ): Gradient[X, DNNApproximation] = {
    val xSeq = xySeq.map(_._1)
    val derivatives = xySeq.map { case (x, y) => derivativeFunction(x, y) }
    val derivativeVector = DenseVector(derivatives.toArray)
    val forwardPropagation = this.forwardPropagation(xSeq).init
    val backwarPropagation = this.backwardPropagation(forwardPropagation, derivativeVector)
    val gradients: Seq[Weights] = backwarPropagation
      .zip(weights)
      .map { case (x, weights) =>
        val gradientVector: DenseVector[Double] = x + weights.weights * regularizationCoefficient
        weights.copy(weights = gradientVector)
      }
    Gradient(this.copy(weights = gradients))
  }
  
  def copy(
    featureFunctions: Seq[X => Double] = this.featureFunctions,
    dnnSpec: DNNSpec = this.dnnSpec,
    regularizationCoefficient: Double = this.regularizationCoefficient,
    weights: Seq[Weights] = this.weights
  ): DNNApproximation[X] =
    DNNApproximation(featureFunctions, dnnSpec, regularizationCoefficient, weights)
  
  /**
   * :param forwardPropagation represents the result of forward propagation (without
   * the final output), a sequence of L 2-D np.ndarrays of the DNN.
   * : param derivativeOutput represents the derivative of the objective
   * function with respect to the linear predictor of the final layer.
   *
   * :return: list (of length L+1) of |O_l| x |I_l| 2-D arrays,
   * i.e., same as the type of self.weights.weights
   * This function computes the gradient (with respect to weights) of
   * the objective where the output layer activation function
   * is the canonical link function of the conditional distribution of y|x
   *
   */
  def backwardPropagation(
    forwardPropagation: Seq[DenseMatrix[Double]],
    objetiveDerivative: DenseVector[Double]
  ): Seq[DenseVector[Double]] = {
    val n = objetiveDerivative.size.toDouble
    val weights = weightMatrices
    val f = dnnSpec.hiddenActivationDerivative
    val g = dnnSpec.outputActivationDerivative
    val lastHiddenOutput = forwardPropagation.last
    val lastWeight = weights.last
    val outputDerivative = (lastHiddenOutput * lastWeight).map(g).toDenseVector
    val lastValue: DenseVector[Double] = lastHiddenOutput.t * (outputDerivative *:* objetiveDerivative) / n
    val backwardDerivatives: Seq[(DenseVector[Double], DenseMatrix[Double])] = weights.tail
      .zip(forwardPropagation.tail)
      .zip(forwardPropagation.init)
      .scanRight((lastValue, objetiveDerivative.toDenseMatrix)) {
        case (((w, x2), x1), (_, objDeriv)) =>
          val y1: DenseMatrix[Double] = w * objDeriv
          val y2: DenseMatrix[Double] = x2.t.map(f)
          val y3: DenseMatrix[Double] = y1 *:* y2
          val y = if (dnnSpec.bias) y3(1 to -1, ::) else y3
          val weightDeriv = (y * x1).t / n
          val weightDerivAsVector = weightDeriv.toDenseVector
          (weightDerivAsVector, y)
      }
    
    backwardDerivatives.map(_._1)
  }
  
  override def updateWithEligibilityTrace(
    eligibilityTrace: Gradient[X, DNNApproximation],
    x: X,
    y: Double
  ): DNNApproximation[X] = {
    val f = eligibilityTrace.functionApproximation
    val error = this.apply(x) - y
    updateWithGradient(Gradient(f * error))
  }
  
  override def apply(x: X): Double = forwardPropagation(Seq(x)).last(0, 0)
  
  /**
   * Update the internal parameters of self FunctionApprox using the
   * input gradient that is presented as a Gradient[FunctionApprox]
   */
  override def updateWithGradient(gradient: Gradient[X, DNNApproximation]): DNNApproximation[X] = {
    val gradientVectors = gradient.functionApproximation.weights
    this.copy(
      weights = this.weights.zip(gradientVectors).map { case (weight, gradientVector) =>
        weight.update(gradientVector.weights)
      }
    )
  }
  
  override def iterateUpdates(
    xySeqStream: Iterator[Iterable[(X, Double)]]
  ): Iterator[DNNApproximation[X]] = {
    IterateUtils.accumulate(
      xySeqStream,
      this)(
      { case (fa, xy) => fa.update(xy) }
    )
  }
  
  private def vectorToMatrix[V](
    vector: DenseVector[V],
    numRows: Int,
    numColumns: Int
  ): DenseMatrix[V] = {
    require(vector.size == numRows * numColumns)
    vector.asDenseMatrix.reshape(numRows, numColumns)
  }
}

object DNNApproximation {
  
  val iterableVectorSpace: VectorSpace[Iterable[Weights]] =
    IterableVectorSpace.iterableVectorSpace(Weights.vectorSpace)
  
  def vectorSpace[X]: VectorSpace[DNNApproximation[X]] =
    new VectorSpace[DNNApproximation[X]] {
      override def add(x: DNNApproximation[X], y: DNNApproximation[X]): DNNApproximation[X] =
        x.copy(weights = iterableVectorSpace.add(x.weights, y.weights).toSeq)
      
      override def multiply(x: DNNApproximation[X], scalar: Double): DNNApproximation[X] =
        x.copy(weights = iterableVectorSpace.multiply(x.weights, scalar).toSeq)
      
      override def norm(x: DNNApproximation[X]): Double =
        iterableVectorSpace.norm(x.weights)
    }
  
  def create[X](
    featureFunctions: Seq[X => Double],
    dnnSpec: DNNSpec,
    adamGradient: AdamGradient = defaultSettings,
    regularizationCoefficient: Double = 0.0,
    weightsOption: Option[Seq[Weights]] = None
  ): DNNApproximation[X] = {
    val weights: Seq[Weights] = weightsOption.getOrElse(
      getInputOutputDimensions(featureFunctions, dnnSpec).map {
        case (input, output) =>
          val randomWeights = DenseVector.rand(input * output, distributions.Gaussian(0.0, 1.0))
          Weights.create(
            weights = randomWeights / sqrt(input),
            adamGradient = adamGradient
          )
      }
    )
    
    apply(
      featureFunctions,
      dnnSpec,
      regularizationCoefficient,
      weights
    )
  }
  
  def apply[X](
    featureFunctions: Seq[X => Double],
    dnnSpec: DNNSpec,
    regularizationCoefficient: Double,
    weights: Seq[Weights]
  ): DNNApproximation[X] =
    new DNNApproximation(featureFunctions, dnnSpec, regularizationCoefficient, weights)
  
  def getInputOutputDimensions[X](
    featureFunctions: Seq[X => Double],
    dnnSpec: DNNSpec
  ): Seq[(Int, Int)] = {
    val inputs = featureFunctions.size +: dnnSpec.neurons.map(_ + (if (dnnSpec.bias) 1 else 0))
    val outputs = dnnSpec.neurons :+ 1
    inputs.zip(outputs)
  }
}

object FunctionApproximationApp extends App {
  
  Locale.setDefault(Locale.US) // To print numbers in US format
  type Triple = (Double, Double, Double)
  val logger: Logger = Logger("FunctionApproximationApp")
  val alpha = 2.0
  val beta1 = 10.0
  val beta2 = 4.0
  val beta3 = -6.0
  val beta = (beta1, beta2, beta3)
  
  val values = DenseVector.rangeD(-10.0, 10.5, 0.5).valuesIterator.toSeq
  val points: Seq[Triple] = for {
    x <- values
    y <- values
    z <- values
  } yield (x, y, z)
  
  val noiseDistribution: Gaussian = Gaussian(sigma = 2.0)
  val xySeq: Seq[(Triple, Double)] = points.map { point =>
    point -> (testFunction(point) + noiseDistribution.sample)
  }
  val adamGradient: AdamGradient = AdamGradient(
    learningRate = 0.1,
    decay1 = 0.9,
    decay2 = 0.999
  )
  val featureFunctions: Seq[Triple => Double] = Seq(
    _ => 1.0,
    x => x._1,
    x => x._2,
    x => x._3
  )
  val linearModel = LinearFunctionApproximation.create(
    featureFunctions = featureFunctions,
    adamGradient = adamGradient,
    regularizationCoefficient = 0.001)
  val directSolveLFA = linearModel.solve(xySeq)
  val directSolveWeights = directSolveLFA.weights
  val directSolveRMSE = directSolveLFA.rmse(xySeq)
  val sgdLFA = linearModel.copy(directSolve = false).solve(xySeq)
  val sdgWeights = sgdLFA.weights
  
  logger.info(f"Linear Model Direct Solve Weights = ${directSolveWeights.weights}\n-----------------------------")
  logger.info(f"Linear Model Direct Solve RMSE = $directSolveRMSE%1.6f\n-----------------------------")
  val sdgRMSE = sgdLFA.rmse(xySeq)
  val dataStream = LazyList.continually(xySeq)
  val weightDiff = directSolveWeights.distanceTo(sdgWeights)
  
  logger.info(f"Linear Model Direct Solve Weights = ${sdgWeights.weights}\n-----------------------------")
  logger.info(f"Linear Model Direct Solve RMSE = $sdgRMSE%1.6f\n-----------------------------")
  val iterations = 150
  logger.info(f"Linear Model: Difference between weights: $weightDiff%1.6f")
  val dnnSpec = DNNSpec(
    neurons = Seq(2),
    bias = true,
    hiddenActivation = identity,
    hiddenActivationDerivative = identity_deriv,
    outputActivation = identity,
    outputActivationDerivative = identity_deriv,
  )
  val dnnApprox = DNNApproximation.create(
    featureFunctions = featureFunctions,
    dnnSpec = dnnSpec,
    adamGradient = adamGradient,
    regularizationCoefficient = 0.01
  )
  val dnnModels = dnnApprox.iterateUpdates(dataStream).take(iterations + 1)
  
  def testFunction(point: Triple): Double = {
    val (x, y, z) = point
    alpha + beta1 * x + beta2 * y + beta3 * z
  }
  
  def identity(x: Double): Double = x
  def identity_deriv(x: Double): Double = 1.0
  
  dnnModels
    .zipWithIndex
    .foreach {
      case (f, i) =>
        val testError = f.rmse(xySeq)
        logger.info(f"Iteration $i - DNN Model - Train RMSE: $testError%1.4f")
    }
  
}