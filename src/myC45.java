import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;

import java.util.Enumeration;

public class myC45 extends Classifier {

    private myNode root = null;

    public myC45() {

    }

    @Override
    public void buildClassifier(Instances data) {
        root = new myNode();
        root.buildClassifier(data);

        prune();
    }

    @Override
    public double classifyInstance(Instance instance) {
        if (root == null) {
            throw new IllegalStateException("Classifier has not build yet");
        }

        return  root.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        if (root == null) {
            throw new IllegalStateException("Classifier has not build yet");
        }

        return root.distributionForInstance(instance);
    }

    /**
     * method to post prune the model
     */
    public void prune() {
        root.prune();
    }
}
