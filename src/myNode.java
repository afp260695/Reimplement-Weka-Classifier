import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.List;

public class myNode extends Classifier {

    private final List<myNode> children = new ArrayList<>();
    private myModel splitedModel;
    private Instances trainingInstances;
    private boolean isLeaf = false;

    private static final double ERROR_THRESHOLD = 0.1;

    @Override
    public void buildClassifier(Instances instances) {
        trainingInstances = instances;
        splitedModel = myModel.chooseFromInstances(instances);
        List<Instances> splitInstances = splitedModel.splitInstances(instances);

        children.clear();

        if (splitInstances.size() == 1) {
            isLeaf = true;
        } else {
            for (Instances childInstances : splitInstances) {
                myNode child = new myNode();
                child.buildClassifier(childInstances);
                children.add(child);
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) {
        double maxProbability = -1;
        int maxIndex = -1;

        for (int i = 0; i < instance.numClasses(); ++i) {
            double probability = calculateProbability(i, instance, 1);

            if (Utils.gr(probability, maxProbability)) {
                maxIndex = i;
                maxProbability = probability;
            }
        }

        return (double)maxIndex;
    }

    /**
     * calculate the probability of an instanse
     */
    private double calculateProbability(int classIdx, Instance instance, double weight) {
        if (isLeaf) {
            return weight * splitedModel.getClassProbability(classIdx);
        } else {
            int idx = splitedModel.determineSubset(instance);

            if (idx < 0) {
                double prob = 0;
                List<Double> weights = splitedModel.getWeights();

                for (int i = 0; i < weights.size(); ++i) {
                    prob += children.get(i).calculateProbability(classIdx, instance, weight * weights.get(i));
                }

                return prob;

            } else {
                return children.get(idx).calculateProbability(classIdx, instance, weight);
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        int numClasses = instance.numClasses();
        double[] result = new double[numClasses];

        for (int i = 0; i < numClasses; ++i) {
            result[i] = calculateProbability(i, instance, 1);
        }

        return result;
    }

    /**
     * Post-prune the tree model
     */
    public void prune() {
        if (isLeaf) {
            // no need to prune anymore
        } else {
            for (myNode child : children) {
                child.prune();
            }

            double errorAsLeaf = calculateErrorAsLeaf();
            double error = calculateError();

            if (Utils.smOrEq(errorAsLeaf, error + ERROR_THRESHOLD)) {
                splitedModel = myModel.createNoSplitModel(trainingInstances);
                children.clear();

                isLeaf = true;
            }
        }
    }

    /**
     * Calculate the error of the tree model recursively.
     */
    private double calculateError() {
        if (isLeaf) {
            return calculateErrorAsLeaf();
        } else {
            double error = 0;

            for (myNode child : children) {
                error += child.calculateError();
            }

            return error;
        }
    }

    /**
     * Calculate the error of a leaf node.
     */
    private double calculateErrorAsLeaf() {
        myModel leafModel = myModel.createNoSplitModel(trainingInstances);
        double totalDistribution = leafModel.getTotalDistribution();

        if (Utils.eq(totalDistribution, 0)) {
            return 0;
        } else {
            double incorrect = 0;
            double dominantClass = leafModel.getDominantClass();
            int numInstances = trainingInstances.numInstances();

            for (int i = 0; i < numInstances; ++i) {
                Instance instance = trainingInstances.instance(i);

                if (!Utils.eq(instance.classValue(), dominantClass)) {
                    incorrect += 1;
                }
            }

            return incorrect / totalDistribution;
        }
    }
}
