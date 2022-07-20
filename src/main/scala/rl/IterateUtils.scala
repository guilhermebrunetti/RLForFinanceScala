package rl

import java.util.Locale

import com.typesafe.scalalogging.Logger
import rl.IterateUtils._

import scala.collection.BufferedIterator

object IterateUtils {
  
  /**
   * Find the fixed point of a function f by applying it to its own
   * result, yielding each intermediate value.
   * That is, for a function f, iterate(f, x) will give us a generator
   * producing:
   * x, f(x), f(f(x)), f(f(f(x)))...
   */
  def iterateLazyList[X](step: X => X, initialValue: X): LazyList[X] = {
    LazyList.unfold(initialValue)((x: X) => Some((x, step(x))))
  }
  
  def iterate[X](step: X => X, initialValue: X): Iterator[X] = {
    Iterator.unfold(initialValue)((x: X) => Some((x, step(x))))
  }
  
  /**
   * Return the final value of the given iterator when its values
   * converge according to the done function.
   * Raises an error if the iterator is empty.
   * Will loop forever if the input iterator doesn't end *or* converge.
   */
  def converged[X](values: Iterable[X], done: (X, X) => Boolean): X = {
    val resultOpt = last(converge(values, done))
    resultOpt.getOrElse(
      throw new IllegalArgumentException(s"Method converged returned None")
    )
  }
  
  /**
   * Return the last value of the given iterator.
   * Returns None if the iterator is empty.
   * If the iterator does not end, this function will loop forever.
   */
  def last[X](values: Iterable[X]): Option[X] = values.lastOption
  
  /**
   * Read from an iterator until two consecutive values satisfy the
   * given done function or the input iterator ends.
   * Raises an error if the input iterator is empty.
   * Will loop forever if the input iterator doesn't end *or* converge.
   */
  def converge[X](values: Iterable[X], done: (X, X) => Boolean): List[X] = {
    val consecutiveValues: Iterable[(X, X)] = values.zip(values.tail)
    val (xs1, xs2) = consecutiveValues.span { case (a, b) => !done(a, b) }
    val (firstList, secondList) = xs1.toList.unzip
    val lastValue = List(xs2.headOption.map(_._2)).flatten
    firstList.head :: secondList ::: lastValue
  }
  
  def converged[X](values: Iterator[X], done: (X, X) => Boolean): X = {
    val converged = converge(values, done)
    if (converged.hasNext) {
      converged.next()
    }
    else {
      throw new IllegalArgumentException(s"Method converge returned empty iterator")
    }
  }
  
  /**
   * Read from an iterator until two consecutive values satisfy the
   * given done function or the input iterator ends.
   * Raises an error if the input iterator is empty.
   * Will loop forever if the input iterator doesn't end *or* converge.
   */
  def converge[X](values: Iterator[X], done: (X, X) => Boolean): Iterator[X] = {
    val bufferedIterator: BufferedIterator[X] = values.buffered
    
    def check(): Boolean = {
      val b = if (bufferedIterator.hasNext) {
        val next = bufferedIterator.next
        !bufferedIterator.headOption.exists(done(next, _))
      } else false
      b
    }
    
    while (check()) {}
    bufferedIterator
  }
  
  /**
   * Make an iterator that returns accumulated sums, or accumulated
   * results of other binary functions (specified via the optional func
   * argument).
   * The number of elements output has one more element than the input iterable.
   */
  def accumulate[X, Y](
    values: Iterable[X],
    initialValue: Y
  )(f: (Y, X) => Y): Iterable[Y] = {
    values.scanLeft(initialValue)(f)
  }
  
  def accumulate[X, Y](
    values: Iterator[X],
    initialValue: Y
  )(f: (Y, X) => Y): Iterator[Y] = {
    values.scanLeft(initialValue)(f)
  }
  
  def accumulateRight[X, Y](
    values: Iterable[X],
    initialValue: Y
  )(f: (X, Y) => Y): Iterable[Y] = {
    values.scanRight(initialValue)(f)
  }
  
}

object IterateApp extends App {
  
  val logger: Logger = Logger("IterateApp")
  Locale.setDefault(Locale.US) // To print numbers in US format
  
  val initialValue = 0.0
  val eps = 1.0e-5
  
  val values: Seq[Double] = converge(
    iterateLazyList((x: Double) => math.cos(x), initialValue),
    (x: Double, y: Double) => math.abs(x - y) < eps
  )
  
  val lastValue = last(values).get
  
  val lastValue2 = converged(
    iterate((x: Double) => math.cos(x), initialValue),
    (x: Double, y: Double) => math.abs(x - y) < eps
  )
  
  val string = values.toVector.zipWithIndex.map { case (f, i) => f"$i: $f%1.8f" }.mkString("\n")
  
  logger.info(s"Iterations:\n$string")
  
  logger.info(s"Numeric solution: $lastValue")
  
  logger.info(s"Alternative Numeric solution: $lastValue2")
}