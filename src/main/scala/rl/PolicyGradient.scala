package rl

import rl.ApproximateDynamicProgramming.{NTStateDistribution, QValueFunctionApproximation, ValueFunctionApproximation}
import rl.utils.{Distribution, Gaussian}


class GaussianPolicyFromApproximation[S](
  val functionApproximation: FunctionApproximation[NonTerminal[S]],
  val sigma: Double
) extends Policy[S, Double] {
  override def act(state: NonTerminal[S]): Distribution[Double] = Gaussian(
    mu = functionApproximation(state),
    sigma = sigma
  )
}

object GaussianPolicyFromApproximation {
  
  def apply[S](
    functionApproximation: FunctionApproximation[NonTerminal[S]],
    sigma: Double
  ): GaussianPolicyFromApproximation[S] =
    new GaussianPolicyFromApproximation(functionApproximation, sigma)
}


object PolicyGradient {
  
  type ActorCriticApproximation[S] =
    (FunctionApproximation[NonTerminal[S]], QValueFunctionApproximation[S, Double])
  type ActorCriticAdvantageApproximation[S] =
    (FunctionApproximation[NonTerminal[S]], QValueFunctionApproximation[S, Double], ValueFunctionApproximation[S])
  type ActorCriticTDErrorApproximation[S] =
    (FunctionApproximation[NonTerminal[S]], ValueFunctionApproximation[S])
  
  def reinforceGaussian[S](
    markovDecisionProcess: MarkovDecisionProcess[S, Double],
    policyMeanApproximation: FunctionApproximation[NonTerminal[S]],
    initialStateDistribution: NTStateDistribution[S],
    policySigma: Double,
    gamma: Double,
    episodeLengthTolerance: Double
  ): Iterator[FunctionApproximation[NonTerminal[S]]] = {
    
    val variance = policySigma * policySigma
    
    Iterator.unfold(policyMeanApproximation) { phi =>
      val policy = GaussianPolicyFromApproximation(phi, policySigma)
      val trace = markovDecisionProcess.simulateAction(initialStateDistribution, policy)
      val returns = Returns.returnsMDP(trace, gamma, episodeLengthTolerance)
      val (newPhi, _) = returns.foldLeft((phi, 1.0)) { case ((funcApprox, gammaProd), step) =>
        
        val grad: Gradient[NonTerminal[S], FunctionApproximation] = FunctionApproximationUtils.objectiveGradient(
          f = funcApprox,
          xySeq = Seq(step.state -> step.action),
          derivativeFunction = (x: NonTerminal[S], y: Double) => (funcApprox(x) - y) / variance
        )
        
        val scaledGrad: Gradient[NonTerminal[S], FunctionApproximation] =
          FunctionApproximationUtils.multiply(grad, gammaProd * step.returns)
        
        val newFuncApprox: FunctionApproximation[NonTerminal[S]] =
          FunctionApproximationUtils.updateWithGradient(funcApprox, scaledGrad)
        
        (newFuncApprox, gammaProd * gamma)
      }
      
      Some((phi, newPhi))
    }
  }
  
  def actorCriticGaussian[S](
    markovDecisionProcess: MarkovDecisionProcess[S, Double],
    policyMeanApproximation: FunctionApproximation[NonTerminal[S]],
    qValueFunctionApproximation: QValueFunctionApproximation[S, Double],
    initialStateDistribution: NTStateDistribution[S],
    policySigma: Double,
    gamma: Double,
    maxEpisodeLength: Int
  ): Iterator[ActorCriticApproximation[S]] = {
    
    def getAction(
      state: NonTerminal[S],
      policyMean: FunctionApproximation[NonTerminal[S]]
    ): Double = {
      Gaussian(mu = policyMean(state), sigma = policySigma).sample
    }
    
    val variance = policySigma * policySigma
    val initialState = initialStateDistribution.sample
    val initialAction = getAction(initialState, policyMeanApproximation)
    val initialArgument = (policyMeanApproximation, qValueFunctionApproximation, initialState, initialAction, 0, 1.0)
    
    Iterator.unfold(initialArgument) { case (phi, q, state, action, step, gammaProd) =>
      val (newState, reward) = markovDecisionProcess.step(state, action).sample
      val stateActionPair = (state, action)
      val (nextQ, nextState, nextAction, nextStep, nextGammaProd) = newState match {
        case nt: NonTerminal[S] if step < maxEpisodeLength =>
          val newAction = getAction(nt, phi)
          val target = reward + gamma * q((nt, newAction))
          val newQ = q.update(Seq(stateActionPair -> target))
          (newQ, nt, newAction, step + 1, gammaProd * gamma)
        case _ =>
          val newQ = q.update(Seq(stateActionPair -> reward))
          val newInitialState = initialStateDistribution.sample
          val newAction = getAction(newInitialState, phi)
          (newQ, newInitialState, newAction, 0, 1.0)
      }
      
      val grad: Gradient[NonTerminal[S], FunctionApproximation] = FunctionApproximationUtils.objectiveGradient(
        f = phi,
        xySeq = Seq((state, action)),
        derivativeFunction = (x: NonTerminal[S], y: Double) => (phi(x) - y) / variance
      )
      
      val scaledGrad: Gradient[NonTerminal[S], FunctionApproximation] =
        FunctionApproximationUtils.multiply(grad, gammaProd * nextQ(stateActionPair))
      
      val nextPhi: FunctionApproximation[NonTerminal[S]] =
        FunctionApproximationUtils.updateWithGradient(phi, scaledGrad)
      
      val nextArgument = (nextPhi, nextQ, nextState, nextAction, nextStep, nextGammaProd)
      Some((phi, q), nextArgument)
    }
  }
  
  def actorCriticAdvantageGaussian[S](
    markovDecisionProcess: MarkovDecisionProcess[S, Double],
    policyMeanApproximation: FunctionApproximation[NonTerminal[S]],
    qValueFunctionApproximation: QValueFunctionApproximation[S, Double],
    valueFunctionApproximation: ValueFunctionApproximation[S],
    initialStateDistribution: NTStateDistribution[S],
    policySigma: Double,
    gamma: Double,
    maxEpisodeLength: Int
  ): Iterator[ActorCriticAdvantageApproximation[S]] = {
    
    def getAction(
      state: NonTerminal[S],
      policyMean: FunctionApproximation[NonTerminal[S]]
    ): Double = {
      Gaussian(mu = policyMean(state), sigma = policySigma).sample
    }
    
    val variance = policySigma * policySigma
    val initialState = initialStateDistribution.sample
    val initialAction = getAction(initialState, policyMeanApproximation)
    val initialArgument =
      (policyMeanApproximation, qValueFunctionApproximation, valueFunctionApproximation, initialState, initialAction, 0, 1.0)
    
    Iterator.unfold(initialArgument) { case (phi, q, v, state, action, step, gammaProd) =>
      val (newState, reward) = markovDecisionProcess.step(state, action).sample
      val stateActionPair = (state, action)
      val (nextQ, nextV, nextState, nextAction, nextStep, nextGammaProd) = newState match {
        case nt: NonTerminal[S] if step < maxEpisodeLength =>
          val newAction = getAction(nt, phi)
          val targetQ = reward + gamma * q(nt -> newAction)
          val targetV = reward + gamma * v(nt)
          val newQ = q.update(Seq(stateActionPair -> targetQ))
          val newV = v.update(Seq(state -> targetV))
          (newQ, newV, nt, newAction, step + 1, gammaProd * gamma)
        case _ =>
          val newQ = q.update(Seq(stateActionPair -> reward))
          val newV = v.update(Seq(state -> reward))
          val newInitialState = initialStateDistribution.sample
          val newAction = getAction(newInitialState, phi)
          (newQ, newV, newInitialState, newAction, 0, 1.0)
      }
      
      val grad: Gradient[NonTerminal[S], FunctionApproximation] = FunctionApproximationUtils.objectiveGradient(
        f = phi,
        xySeq = Seq((state, action)),
        derivativeFunction = (x: NonTerminal[S], y: Double) => (phi(x) - y) / variance
      )
      
      val scaledGrad: Gradient[NonTerminal[S], FunctionApproximation] =
        FunctionApproximationUtils.multiply(grad, gammaProd * (nextQ(stateActionPair) - nextV(state)))
      
      val nextPhi: FunctionApproximation[NonTerminal[S]] =
        FunctionApproximationUtils.updateWithGradient(phi, scaledGrad)
      
      val nextArgument = (nextPhi, nextQ, nextV, nextState, nextAction, nextStep, nextGammaProd)
      Some((phi, q, v), nextArgument)
    }
  }
  
  def actorCriticTDErrorGaussian[S](
    markovDecisionProcess: MarkovDecisionProcess[S, Double],
    policyMeanApproximation: FunctionApproximation[NonTerminal[S]],
    valueFunctionApproximation: ValueFunctionApproximation[S],
    initialStateDistribution: NTStateDistribution[S],
    policySigma: Double,
    gamma: Double,
    maxEpisodeLength: Int
  ): Iterator[ActorCriticTDErrorApproximation[S]] = {
    
    def getAction(
      state: NonTerminal[S],
      policyMean: FunctionApproximation[NonTerminal[S]]
    ): Double = {
      Gaussian(mu = policyMean(state), sigma = policySigma).sample
    }
    
    val variance = policySigma * policySigma
    val initialState = initialStateDistribution.sample
    val initialArgument =
      (policyMeanApproximation, valueFunctionApproximation, initialState, 0, 1.0)
    
    Iterator.unfold(initialArgument) { case (phi, v, state, step, gammaProd) =>
      val action = getAction(state, phi)
      val (newState, reward) = markovDecisionProcess.step(state, action).sample
      val (tdTarget, nextState, nextStep, nextGammaProd) = newState match {
        case nt: NonTerminal[S] if step < maxEpisodeLength =>
          (reward + gamma * v(nt), nt, step + 1, gammaProd * gamma)
        case _ =>
          val newInitialState = initialStateDistribution.sample
          (reward, newInitialState, 0, 1.0)
      }
      val tdError = tdTarget - v(state)
      val nextV = v.update(Seq(state -> tdTarget))
      
      val grad: Gradient[NonTerminal[S], FunctionApproximation] = FunctionApproximationUtils.objectiveGradient(
        f = phi,
        xySeq = Seq((state, action)),
        derivativeFunction = (x: NonTerminal[S], y: Double) => (phi(x) - y) / variance
      )
      
      val scaledGrad: Gradient[NonTerminal[S], FunctionApproximation] =
        FunctionApproximationUtils.multiply(grad, gammaProd * tdError)
      
      val nextPhi: FunctionApproximation[NonTerminal[S]] =
        FunctionApproximationUtils.updateWithGradient(phi, scaledGrad)
      
      val nextArgument = (nextPhi, nextV, nextState, nextStep, nextGammaProd)
      Some((phi, v), nextArgument)
    }
  }
  
  
}
