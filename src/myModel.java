import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.*;

public class myModel extends Classifier {

    private final int splitAttributeIndex;
    private int numValues;
    private int numClasses;
    private double splitPoint = Double.MAX_VALUE;

    private double gainRatio;
    private double infoGain;

    private double[][] distribution;
    private double[] classDistribution;
    private double[] valueDistribution;

    private List<Double> weights;
    private boolean built = false;
    private boolean noSplit = false;


    public myModel(int splitAttributeIndex) {
        this.splitAttributeIndex = splitAttributeIndex;
    }

    /**
     * create a tree model without split
     */
    public static myModel createNoSplitModel(Instances instances) {
        myModel model = new myModel(-1);
        model.noSplit = true;
        model.buildClassifier(instances);

        return model;
    }

    /**
     * Create a tree model with the selection of attributes using
     * gain ratio.
     */
    public static myModel chooseFromInstances(Instances instances) {
        double maxGainRatio = 0;
        myModel bestModel = null;

        for (int i = 0; i < instances.numAttributes(); ++i) {
            if (i == instances.classIndex()) {
                continue;
            }

            myModel model = new myModel(i);
            model.buildClassifier(instances);

            double gainRatio = model.getGainRatio();

            if (Utils.gr(gainRatio, maxGainRatio)) {
                maxGainRatio = gainRatio;
                bestModel = model;
            }
        }

        if (bestModel == null) {
            return createNoSplitModel(instances);
        } else {
            return bestModel;
        }
    }

    @Override
    public void buildClassifier(Instances instances) {
        if (noSplit) {
            buildNoSplit(instances);
        } else {
            build(instances);
        }
        built = true;
    }

    /**
     * method to build a tree with out division.
     */
    private void buildNoSplit(Instances instances) {
        numClasses = instances.numClasses();
        numValues = 1;

        distribution = new double[numValues][numClasses];
        classDistribution = new double[numClasses];
        valueDistribution = new double[numValues];

        //calculate the total appearance of each class value
        for (int i = 0; i < instances.numInstances(); ++i) {
            Instance instance = instances.instance(i);
            distribution[0][(int)instance.classValue()] += 1;
        }

        //calculate the total appearance of
        for (int i = 0; i < numValues; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                classDistribution[j] += distribution[i][j];
                valueDistribution[i] += distribution[i][j];
            }
        }

        weights = new ArrayList<>();
        weights.add(0.);

        infoGain = 0;
        gainRatio = 0;
    }

    /**
     * method to build a tree model
     */
    private void build(Instances instances) {
        Attribute splitAttribute = instances.attribute(splitAttributeIndex);
        boolean isNominal = splitAttribute.isNominal();

        numClasses = instances.numClasses();
        numValues = isNominal ? instances.numDistinctValues(splitAttributeIndex) : 2;

        distribution = new double[numValues][numClasses];
        classDistribution = new double[numClasses];
        valueDistribution = new double[numValues];

        double total = 0;

        if (!isNominal) {
            splitPoint = determineSplitPoint(instances);
        }

        //calculate the appearance of certain class value in an attribut
        for (int i = 0; i < instances.numInstances(); ++i) {
            Instance instance = instances.instance(i);

            if (!instance.isMissing(splitAttributeIndex)) {
                resolveMissingValue(instances);
            }

            if (isNominal) {
                distribution[(int)instance.value(splitAttributeIndex)][(int)instance.classValue()] += 1;
            } else { //numerik
                int value = Utils.gr(splitPoint, instance.value(splitAttributeIndex)) ? 0 : 1;
                distribution[value][(int)instance.classValue()] += 1;
            }
        }

        for (int i = 0; i < numValues; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                classDistribution[j] += distribution[i][j];
                valueDistribution[i] += distribution[i][j];
                total += distribution[i][j];
            }
        }

        weights = new ArrayList<>();

        if (Utils.eq(total, 0)) {
            for (int i = 0; i < numValues; ++i) {
                weights.add(0.);
            }
        } else {
            for (int i = 0; i < numValues; ++i) {
                weights.add(valueDistribution[i] / total);
            }
        }

        //info gain from splited attribute
        infoGain = calculateEntropy(classDistribution);

        //info gain every values in splited attribute
        for (int i = 0; i < numValues; ++i) {
            infoGain -= (valueDistribution[i] / total) * calculateEntropy(distribution[i]);
        }

        double splitInfo = 0;

        //SplitInformation(S, A)
        for (int i = 0; i < numValues; ++i) {
            double ratio = valueDistribution[i] / total;

            if (Utils.gr(ratio, 0)) {
                splitInfo -= ratio * Utils.log2(ratio);
            }
        }

        gainRatio = Utils.eq(splitInfo, 0) ? infoGain : (infoGain / splitInfo);
    }

    /**
     * method to choose splitting point to process numeric data
     */
    private double determineSplitPoint(Instances instances) {
        instances.sort(splitAttributeIndex);

        double lastValue = Double.MIN_VALUE;
        double lastClass = -1;

        for (int i = 0; i < instances.numInstances(); ++i) {
            Instance instance = instances.instance(i);

            if (!instance.isMissing(splitAttributeIndex)) {
                resolveMissingValue(instances);
            }

            double currentClass = instance.classValue();
            double currentValue = instance.value(splitAttributeIndex);

            if (!Utils.eq(currentClass, lastClass)) {
                if (lastClass == -1) {
                    lastClass = currentClass;
                } else {
                    return (currentValue + lastValue) / 2;
                }
            }
            lastValue = currentValue;
        }

        return Double.MAX_VALUE;
    }

    /**
     * calculate the entropy of an attributes
     */
    public static double calculateEntropy(double[] values) {
        double total = 0;

        for (double value : values) {
            total += value;
        }

        if (Utils.eq(total, 0)) {
            return 0;
        }

        double entropy = 0;

        for (double value : values) {
            double p = value / total;

            if (Utils.gr(p, 0)) {
                entropy -= p * Utils.log2(p);
            }
        }

        return entropy;
    }

    /**
     * split the instances to make a tree model
     */
    public List<Instances> splitInstances(Instances instances) {
        if (!built) {
            throw new IllegalStateException("Model has not built yet");
        }

        List<Instances> result = new ArrayList<>();
        int numInstances = instances.numInstances();

        for (int i = 0; i < numValues; ++i) {
            result.add(new Instances(instances, numInstances));
        }

        for (int i = 0; i < numInstances; ++i) {
            Instance instance = instances.instance(i);
            int subset = determineSubset(instance);

            if (subset == -1) {
                for (int j = 0; j < numValues; ++j) {
                    double weight = weights.get(i);

                    if (Utils.gr(weight, 0)) {
                        result.get(j).add(instance);
                        result.get(j).lastInstance().setWeight(weight * instance.weight());
                    }
                }
            } else {
                result.get(subset).add(instance);
            }
        }
        return result;
    }

    /**
     * categorize the type of instances
     */
    public int determineSubset(Instance instance) {
        if (!built) {
            throw new IllegalStateException("Model has not built yet");
        }

        if (noSplit) {
            return 0;
        } else if (instance.isMissing(splitAttributeIndex)) {
            return -1;
        } else if (instance.attribute(splitAttributeIndex).isNominal()) {
            return (int)instance.value(splitAttributeIndex);
        } else {
            return Utils.gr(splitPoint, instance.value(splitAttributeIndex)) ? 0 : 1;
        }
    }

    /**
     * return the value of propability based on class value
     */
    public double getClassProbability(int classIndex) {
        if (!built) {
            throw new IllegalStateException("Model has not built yet");
        }

        double total = getTotalDistribution();

        if (Utils.eq(total, 0)) {
            return 0;
        } else {
            return classDistribution[classIndex] / total;
        }
    }

    /**
     * return the weight of the model
     */
    public List<Double> getWeights() {
        if (!built) {
            throw new IllegalStateException("Model has not built yet");
        }

        return weights;
    }

    /**
     * return the gain ration of the model
     */
    public double getGainRatio() {
        if (!built) {
            throw new IllegalStateException("Model has not built yet");
        }

        return gainRatio;
    }

    /**
     * calculate the total distribution of every value
     */
    public double getTotalDistribution() {
        if (!built) {
            throw new IllegalStateException("Model has not built yet");
        }

        double total = 0;
        for (int i = 0; i < numClasses; ++i) {
            total += classDistribution[i];
        }
        return total;
    }

    /**
     * return the total amount of classes value
     */
    public int getNumClasses() {
        if (!built) {
            throw new IllegalStateException("Model has not built yet");
        }

        return numClasses;
    }

    /**
     * return the most dominant class by calculate the
     * probability
     */
    public double getDominantClass() {
        int dominantClass = 0;
        double maxProbability = 0;

        for (int i = 0; i < numClasses; ++i) {
            double classProbability = getClassProbability(i);

            if (classProbability > maxProbability) {
                maxProbability = classProbability;
                dominantClass = i;
            }
        }

        return dominantClass;
    }

    /**
     * set the missing attribute with most common
     * value attribute with the same class
     */
    public void resolveMissingValue(Instances instances) {
        for (int i = 0; i < instances.numInstances(); ++i) {
            Instance instance = instances.instance(i);

            for (int j = 0; j < instance.numAttributes() - 1; ++j) {
                if (instance.isMissing(j)) {
                    instance.setValue(j, mostCommonTarget(instances, instance.classValue(), j));
                }
            }
        }
    }


    /**
     * return the value of most common value attribute
     * with the same target
     */
    public double mostCommonTarget(Instances instances, double target, int indexMissing) {

        int numValueAtt = instances.numDistinctValues(indexMissing); //numValues
        int[] distributionAtt = new int[numValueAtt];

        Enumeration enumInstance = instances.enumerateInstances();

        for (int i = 0; i < numValueAtt; ++i) {
            distributionAtt[i] = 0;
        }

        while (enumInstance.hasMoreElements()) {
            Instance instance = (Instance) enumInstance.nextElement();

            if (instance.classValue() == target && !instance.isMissing(indexMissing)) {
                distributionAtt[(int) instance.value(indexMissing)]++;
            }
        }

        int maxIdx = 0;
        int max = 0;

        for (int i = 0; i < numValueAtt; ++i) {
            if (distributionAtt[i] > max) {
                max = distributionAtt[i];
                maxIdx = i;
            }
        }

        return (double) maxIdx;
    }
}
