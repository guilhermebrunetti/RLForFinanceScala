package rl.utils

import scala.math.Numeric

/**
 *
 * Normed VectorSpace for type V
 */
trait VectorSpace[V] {
  def add(x: V, y: V): V
  
  def multiply(x: V, scalar: Double): V
  
  def norm(x: V): Double
  
  def distance(x: V, y: V): Double = norm(subtract(x, y))
  
  def subtract(x: V, y: V): V = add(x, multiply(y, -1))
}

trait FunctionalVectorSpace[X, F[_]] extends VectorSpace[F[X]]

trait VectorSpaceOperators[V] {
  self: V =>
  
  def op: VectorSpace[V]
  
  def +(other: V): V = op.add(self, other)
  
  def *(scalar: Double): V = op.multiply(self, scalar)
  
  def -(other: V): V = op.subtract(self, other)
  
  def norm: Double = op.norm(self)
  
  def within(other: V, tolerance: Double): Boolean = {
    distanceTo(other) <= tolerance
  }
  
  def distanceTo(other: V): Double = op.distance(this, other)
}

object MapAlgebra {
  
  def subtractMap[X, B](xMap: Map[X, B], yMap: Map[X, B])(implicit num: Numeric[B]): Map[X, B] = {
    (xMap.keySet ++ yMap.keySet).map { k =>
      k -> num.minus(xMap.getOrElse(k, num.zero), yMap.getOrElse(k, num.zero))
    }.toMap
  }
  
  def sumMap[X, B](xMap: Map[X, B], yMap: Map[X, B])(implicit num: Numeric[B]): Map[X, B] = {
    (xMap.keySet ++ yMap.keySet).map { k =>
      k -> num.plus(xMap.getOrElse(k, num.zero), yMap.getOrElse(k, num.zero))
    }.toMap
  }
  
  def multiplyByScalar[X, B](xMap: Map[X, B], scalar: B)(implicit num: Numeric[B]): Map[X, B] = {
    xMap.view.mapValues(num.times(_, scalar)).toMap
  }
  
  def productMap[X, B](xMap: Map[X, B], yMap: Map[X, B])(implicit num: Numeric[B]): Map[X, B] = {
    (xMap.keySet ++ yMap.keySet).map { k =>
      k -> num.times(xMap.getOrElse(k, num.zero), yMap.getOrElse(k, num.zero))
    }.toMap
  }
}



object IterableVectorSpace {
  
  def iterableVectorSpace[V](vectorSpace: VectorSpace[V]): VectorSpace[Iterable[V]] = {
    new VectorSpace[Iterable[V]] {
      override def add(x: Iterable[V], y: Iterable[V]): Iterable[V] = {
        require(x.size == y.size, "x and y should have same size")
        x.zip(y).map { case (a, b) => vectorSpace.add(a, b) }
      }
      
      override def multiply(x: Iterable[V], scalar: Double): Iterable[V] =
        x.map(v => vectorSpace.multiply(v, scalar))
      
      override def norm(x: Iterable[V]): Double = x.map(vectorSpace.norm).max
    }
  }
}


