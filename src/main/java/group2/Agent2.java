package group2;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import negotiator.AgentID;
import negotiator.Bid;
import negotiator.Domain;
import negotiator.actions.Accept;
import negotiator.actions.Action;
import negotiator.actions.Offer;
import negotiator.issue.Issue;
import negotiator.issue.IssueDiscrete;
import negotiator.issue.ValueDiscrete;
import negotiator.parties.AbstractNegotiationParty;
import negotiator.parties.NegotiationInfo;
import negotiator.timeline.TimeLineInfo;

public class Agent2 extends AbstractNegotiationParty {
	
	private final String description = "NeuralAgent";
	ComputationGraphConfiguration nn;
	ComputationGraph mlnn;
	double beta = 0.15;
	double target = 1.0;
	double minTarget = 0.7;
	List<HashMap<ValueDiscrete, Integer>> indexMappings = new ArrayList<HashMap<ValueDiscrete, Integer>>();
	boolean exploring = true;
	double explorationTime = 0.5;
	
	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		this.nn = buildNN(info);
		this.exploring = true;
		this.mlnn = new ComputationGraph(this.nn);
		this.buildIndex(info.getUtilitySpace().getDomain());
	}
	
	public ComputationGraphConfiguration buildNN(NegotiationInfo info) {
		Domain dom = info.getUtilitySpace().getDomain();
		NeuralNetConfiguration.Builder nnBuilder = new NeuralNetConfiguration.Builder();
		ComputationGraphConfiguration.GraphBuilder gb = makeGraphBuilder(nnBuilder);
		ComputationGraphConfiguration.GraphBuilder gbi = addInputLayers(dom, gb);
		ComputationGraphConfiguration.GraphBuilder gbh = addHiddenLayers(dom, gbi);
		ComputationGraphConfiguration.GraphBuilder gbo = addOutputLayer(dom, gbh);
		return gbo.build();
	}
	
	public ComputationGraphConfiguration.GraphBuilder makeGraphBuilder(NeuralNetConfiguration.Builder nnBuilder) {
		return nnBuilder
			.weightInit(WeightInit.XAVIER)
			.learningRate(0.01)
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			.iterations(1)
			.graphBuilder();
	}
	
	public ComputationGraphConfiguration.GraphBuilder addInputLayers(Domain dom, ComputationGraphConfiguration.GraphBuilder gb) {
		List<Issue> issues = dom.getIssues();
		String[] inputNames = new String[issues.size()];
		InputType[] inputs = new InputType[issues.size()];
		
		for(int i = 0; i < issues.size(); i++) {
			Issue issue = issues.get(i);
			String iname = issue.getName();
			IssueDiscrete dissue = (IssueDiscrete) issue;
			int numValues = dissue.getNumberOfValues();
			inputNames[i] = iname;
			inputs[i] = InputType.feedForward(numValues);
		}
		
		return gb
				.addInputs(inputNames)
				.setInputTypes(inputs);
	}
	
	public ComputationGraphConfiguration.GraphBuilder addHiddenLayers(Domain dom, ComputationGraphConfiguration.GraphBuilder gb) {
		List<Issue> issues = dom.getIssues();
		ComputationGraphConfiguration.GraphBuilder cgb = gb;
		
		for (int i = 0; i < issues.size(); i++) {
			IssueDiscrete issue = (IssueDiscrete) issues.get(i);
			String iname = issue.getName();
			//can't use getNumber here.... obviously
			cgb = gb.addLayer(iname+"-hidden", new DenseLayer.Builder()
					.nIn(issue.getNumberOfValues())
					.nOut(1)
					.activation("identity")
					.build(), iname);
		}
		
		return cgb;
	}
	
	public ComputationGraphConfiguration.GraphBuilder addOutputLayer(Domain dom, ComputationGraphConfiguration.GraphBuilder gb) {
		String[] hiddenLayerNames = new String[dom.getIssues().size()];
		
		for(int i = 0; i < dom.getIssues().size(); i++) {
			hiddenLayerNames[i] = dom.getIssues().get(i).getName() + "-hidden";
		}
		
		return gb
				.addLayer("output", new OutputLayer.Builder()
					.activation("identity")
					.nIn(dom.getIssues().size())
					.nOut(1)
					.lossFunction(LossFunctions.LossFunction.MSE)
					.build(), hiddenLayerNames)
				.setOutputs("output")
				.pretrain(false)
				.backprop(true);
	}
	
	@Override
	public void receiveMessage(AgentID sender, Action act) {
		super.receiveMessage(sender, act);
		
		if (act instanceof Accept) {
			Accept acceptance = (Accept) act;
			INDArray[] inputs = buildInput(acceptance.getBid());
			double[][] target = {{1}};
			INDArray[] output = {new NDArray(target)};
			MultiDataSet mds = new MultiDataSet(inputs, output);
			mlnn.fit(mds);
		}
		
		else if (act instanceof Offer) {
			Offer offer = (Offer) act;
			INDArray[] inputs = buildInput(offer.getBid());
			double[][] target = {{1}};
			INDArray[] output = {new NDArray(target)};
			MultiDataSet mds = new MultiDataSet(inputs, output);
			mlnn.fit(mds);
		}
	}
	
	public void buildIndex(Domain dom) {
		List<Issue> issues = dom.getIssues();
		
		for(int i = 0; i < issues.size(); i++) {
			HashMap<ValueDiscrete, Integer> mappings = new HashMap<ValueDiscrete, Integer>();
			IssueDiscrete issue = (IssueDiscrete) issues.get(i);
			
			for(int j =0; j < issue.getNumberOfValues(); j++) {
				mappings.put(issue.getValue(j), j);
			}
			
			this.indexMappings.add(mappings);
		}
	}
	
	public NDArray buildOneHot(int issueNo, Bid b) {
		IssueDiscrete issue = (IssueDiscrete) b.getIssues().get(issueNo);
		double[][] vector = new double[1][issue.getNumberOfValues()];
		
		//Changed to getNumber() here...
		vector[0][indexMappings.get(issueNo).get(b.getValue(issue.getNumber()))] = 1.0;
		
		return new NDArray(vector);
	}
	
	public INDArray[] buildInput(Bid b) {
		
		INDArray[] inputs = new INDArray[b.getIssues().size()];
		
		for(int i = 0; i < b.getIssues().size(); i++) {
			inputs[i] = buildOneHot(i, b);
		}
		
		return inputs;
		
	}
	
	//Accept, Offer or EndNegotiation
	public Action chooseAction(List<Class<? extends Action>> list) {
		
		this.updatePhase();
		this.updateTarget();
		Bid lastBid = null;
		Bid newBid = null;
		
		if(this.getLastReceivedAction() != null &&  this.getLastReceivedAction() instanceof Accept) {
			lastBid = ((Accept) this.getLastReceivedAction()).getBid();
		}
		
		else if(this.getLastReceivedAction() != null && this.getLastReceivedAction() instanceof Offer) {
			lastBid = ((Offer) this.getLastReceivedAction()).getBid();
		}
		
		//We know this must return an actual bid
		if (lastBid != null && this.getUtility(lastBid) > this.target) {
			return new Accept(this.getPartyId(), lastBid);
		}
			
		else if(!exploring) {
			double agreementChance = Double.MIN_VALUE;
			TimeLineInfo timeline = this.getTimeLine();
			double start = timeline.getCurrentTime();
				
			do {
				Bid candBid = this.generateRandomBidWithUtility(this.target);
				INDArray[] testInput = this.buildInput(candBid);
				INDArray[] testOutput = this.mlnn.output(testInput);
				
				if(testOutput[0].getDouble(0,0) >= agreementChance) {
					newBid = candBid;
					agreementChance = testOutput[0].getDouble(0,0);
				}
			} while (agreementChance < 1.0 && timeline.getCurrentTime() < start + 3.0);
			
			return new Offer(this.getPartyId(), newBid);
		}
		
		//We know this must set newBid properly
		else {
			newBid = this.generateRandomBidWithUtility(this.target);
			return new Offer(this.getPartyId(), newBid);
		}
			
		//return new Offer(this.getPartyId(), newBid);
	}
	
	public void updateTarget() {
		try {
			
			TimeLineInfo timeline = this.getTimeLine();
			double t = timeline.getTime();
			Bid bidMax = this.utilitySpace.getMaxUtilityBid();
			Bid bidMin = this.utilitySpace.getMinUtilityBid();
			
			if(exploring) {
				double ft = Math.pow(t, 1.0/beta);
				this.target = getUtility(bidMax) - ft*(getUtility(bidMax) - getUtility(bidMin));
			}
			
			else {
				double ft = Math.pow(t/this.explorationTime, 1.0/beta);
				this.target = getUtility(bidMax) - ft*(getUtility(bidMax) - getUtility(bidMin));
			}
			
			if (this.target < this.minTarget) {
				this.target = this.minTarget;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void updatePhase() {
		TimeLineInfo timeline = this.getTimeLine();
		double t = timeline.getTime();
			
		if (t > this.explorationTime) {
			exploring  = false;
		}
	}
	
	public Bid generateRandomBidWithUtility(double utilityThreshold) {
	      Bid randomBid;
	      double utility;
	      do {
	          randomBid = generateRandomBid();
	          try {
	              utility = utilitySpace.getUtility(randomBid);
	          } catch (Exception e)
	          {
	        	  e.printStackTrace();
	              utility = 0.0;
	          }
	      }
	      while (utility < utilityThreshold);
	      return randomBid;
	 }

	public String getDescription() {
		return this.description;
	}
}

