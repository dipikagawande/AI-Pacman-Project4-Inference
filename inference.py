import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined
import util

class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """


        if self.total() == 0.0:
            return

        tot = self.total()
        for key in self.keys():
            self[key] = float(self[key])/tot




    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """


        s_seq = []
        s_weights = []

        for item in self.items():
            s_seq.append(item[0])
            s_weights.append(float(item[1])/float(self.total()))

        x = random.random()

        for i, val in enumerate(s_seq):
            if x<=s_weights[i]:
                return val
            x-=s_weights[i]



class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """

        "*** YOUR CODE HERE ***"

        if noisyDistance == None and jailPosition == ghostPosition:
            return 1
        elif noisyDistance == None and jailPosition != ghostPosition:
            return 0
        elif noisyDistance != None and jailPosition ==ghostPosition:
            return 0


        obs = busters.getObservationProbability(noisyDistance, manhattanDistance(pacmanPosition,ghostPosition))
        return obs

        #raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.
        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        ## beliefs are stored in self.beliefs, which is a sort of dict object
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.
        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.
        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"

        pacmanPos = gameState.getPacmanPosition()
        jailPos = self.getJailPosition()

        ## For each possible ghost position:
        for pos in self.allPositions:
            ## getObservationProb(self, noisyDistance, pacmanPos, ghostPos, jailPos) 
            ## gives conditional P(noisyDistance | pacmanPos, ghostPos).
            condProb = self.getObservationProb(observation, pacmanPos, pos, jailPos)
            ## Update the belief weighting the prior by this conditional prob
            self.beliefs[pos] = condProb * self.beliefs[pos]
            
        self.beliefs.normalize()


    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.

        Your agent has access to the action distribution for the ghost through 
        self.getPositionDistribution. In order to obtain the distribution over 
        new positions for the ghost, given its previous position, use this line 
        of code:

        newPosDist = self.getPositionDistribution(gameState, oldPos)

        Where oldPos refers to the previous ghost position.
        newPosDist is a DiscreteDistribution object, where for each position p 
        in self.allPositions, newPosDist[p] is the probability that the ghost is 
        at position p at time t + 1, given that the ghost is at position oldPos 
        at time t.

        """
        "*** YOUR CODE HERE ***"

        ## Initialize an empty dictionary for the PDF of oldPos
        oldPosPDF = DiscreteDistribution()        
        
        ## For each oldPos:
        for oldPos in self.allPositions:
            ## Get P(oldPos) for this oldPos
            oldPosProb = self.beliefs[oldPos]
            ## Make newPosDist dict, with conditional probs for each newPos:
            ## newPos: P(newPos at t+1 | oldPos at t)
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            ## Under this oldPos, for each newPos:
            for newPos in newPosDist.keys():
                ## Populate the PDF for this oldPos by summing over all newPos i's
                ## P(oldPos) = SUM_i [P(newPos i, oldPos)]
                ##           = SUM_i [P(newPos i | oldPos) * P(oldPos)]
                oldPosPDF[newPos] += newPosDist[newPos] * oldPosProb
            
        self.beliefs = oldPosPDF
        self.beliefs.normalize()


    def getBeliefDistribution(self):
        return self.beliefs

        


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"

        i = 0
        ## Until we run out of particles:
        while i < self.numParticles:
            ## For each possible board position:
            for pos in self.legalPositions:
                ## Append this board position in the slot for particle i (ie,
                ## put particle i in this board position)
                self.particles.append(pos)
                ## Increment to next particle
                i += 1

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"

        pacmanPos = gameState.getPacmanPosition()
        jailPos = self.getJailPosition()
        ## Initialize a conditional PDF for noisyDistance | pacmanPos as a dict
        condPDFNoisyDist = DiscreteDistribution()

        ## For each possible ghost position (particle position):
        for pos in self.particles:
            ## Get conditional P(noisyDistance | pacmanPos, particle position)
            condProb = self.getObservationProb(observation, pacmanPos, pos, jailPos)
            ## Get P(noisyDistance | pacmanPos, possible ghost position) by
            ## summing the conditional probability of noisyDistance for all
            ## particles that are in this position.
            condPDFNoisyDist[pos] += condProb
        
        ## Now we sample from the conditional PDF.
        ## If at least one particle has non-zero weight:
        if condPDFNoisyDist.total () > 0: 
            ## Normalize the conditional PDF and set it as the belief distrib
            self.beliefs = condPDFNoisyDist.normalize()
            ## For each particle i: 
            for i in range(self.numParticles):
                ## Sample from condPDF and return the key (particle position)
                thisParticleSample = condPDFNoisyDist.sample()
                ## Update the ith particle in the list with this sampled 
                ## particle position from the conditional PDF
                self.particles[i] = thisParticleSample
        ## Else, all particles have zero weight:
        else:
            ## Re-initialize the particle list uniformly
            self.initializeUniformly(gameState)
    


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"

        ## Initialize a list of new position for each particle
        newPositions = []

        ## For each current (old) particle position:
        for oldPos in self.particles:
            ## Generate the PDF of this particle's newPos based on its 
            ## current state (oldPos) and the gameState
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            ## Sample this particle's newPos from the PDF and add it to the
            ## new positions list
            newPositions.append(newPosDist.sample())
        
        ## Update the particles list as new Positions
        self.particles = newPositions
    

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        """
        "*** YOUR CODE HERE ***"
        ## Initialize belief distribution
        beliefDist = DiscreteDistribution()

        ## For each possible ghost position (particle position):
        for pos in self.particles:
            ## Add to each particle position, depending on how heavily that
            ## potential ghost position is weighted (ie, how strong the
            ## belief is that a ghost is at that position = number of particles
            ## at that board position)
            beliefDist[pos] += 1
        ## Normalize over the total number of particles
        beliefDist.normalize()

        return beliefDist



class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"

        ## As suggested in the problem, get all possible permutations of
        ## particles/ghost positions (all ordered pairs given by Cartesian product). 
        ## Repeat the list legalPositions for as many ghosts as we have, so
        ## we get a list of tuples with one entry for each ghost.
        ghostPosPerms = list(itertools.product(self.legalPositions, repeat = self.numGhosts))
        random.shuffle(ghostPosPerms)

        ## Initialize particle counter
        i = 0
        ## Until we run out of particles:
        while i < self.numParticles:
            ## Modulus. As long as particle counter i < # of position tuples, 
            ## modulus will be i. When i = number of tuples, we have put
            ## one particle in each position tuple, and modulus is zero.
            ## Then modulus starts incrementing by i again.
            modulus = i % len(ghostPosPerms)
            ## Append position tuple i (or zero depending on modulus) in the 
            ## slot for particle i (ie, put particle i in this position tuple)
            self.particles.append(ghostPosPerms[modulus])
            ## Increment to next particle
            i += 1


    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.

        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Resample particles based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.
        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.

        To loop over all the ghosts, use:
            for i in range(self.numGhosts):

        You can still obtain Pacman’s position using gameState.getPacmanPosition(), but to get the jail
        position for a ghost, use self.getJailPosition(i), since now there are multiple ghosts each with their own jail positions.

        As in the update method for the ParticleFilter class, you should again use the function self.getObservationProb
        to find the probability of an observation given Pacman’s position, a potential ghost position, and the jail position.
        The sample method of the DiscreteDistribution class will also be useful.

        """
        "*** YOUR CODE HERE ***"

        pacmanPos = gameState.getPacmanPosition()
        ## Initialize a conditional PDF for noisyDistance | pacmanPos as a dict
        condPDFNoisyDist = DiscreteDistribution()

        ## For each possible ghost position (particle position):
        for pos in self.particles:
            ## Initialize conditional prob for this particle position, since
            ## we will multiply into it
            condProbThisParticlePos = 1
            ## For each ghost:
            for i in range(self.numGhosts):
                ## Store the noisy observation for this ghost
                obsThisGhost = observation[i]
                ## Store the jail position for this ghost
                jailPosThisGhost = self.getJailPosition(i)
                ## Get conditional P(noisyDistance | pacmanPos, this particle's) 
                ## position) by multiplying the conditional prob for all ghosts
                ## who could be at the location of this particle
                condProbThisParticlePos *= self.getObservationProb(obsThisGhost, 
                                                                   pacmanPos, 
                                                                   pos[i], 
                                                                   jailPosThisGhost)
            ## Get P(noisyDistance | pacmanPos, this particle's position) by
            ## summing the conditional probability of noisyDistance for all
            ## particles that are in this position.
            condPDFNoisyDist[pos] += condProbThisParticlePos

        ## Now we sample from the conditional PDF.
        ## If at least one particle has non-zero weight:
        if condPDFNoisyDist.total() > 0:
            ## Normalize the conditional PDF and set it as the belief distrib
            self.beliefs = condPDFNoisyDist.normalize()
            ## For each particle i: 
            for i in range(self.numParticles):
                ## Sample from condPDF and return the key (particle position)
                thisParticleSample = condPDFNoisyDist.sample()
                ## Update the ith particle in the list with this sampled 
                ## particle position from the conditional PDF
                self.particles[i] = thisParticleSample
        ## Else, all particles have zero weight:
        else:
            ## Re-initialize the particle list uniformly
            self.initializeUniformly(gameState)


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.

        As in the last question, you can loop over the ghosts using:
            for i in range(self.numGhosts):

        Then, assuming that i refers to the index of the ghost, to obtain the 
        distributions over new positions for that single ghost, given the list 
        (prevGhostPositions) of previous positions of all of the ghosts, use:

        newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])

        """
        newParticles = []

        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions, for updating
            
            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            
            # A list of ghost positions, for referencing
            prevGhostPositions = list(oldParticle) 

            ## For each ghost index:
            for i in range(self.numGhosts):
                ## Generate the PDF of this ghost's newPos based on
                ## the gameState and previous positions of all the ghosts
                newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])
                ## Sample this ghost's newPos from the PDF and store as
                ## this ghost's entry in newParticle
                newParticle[i] = newPosDist.sample()
                
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles

        


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist