import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Resample;

import java.util.Random;

public class MainJ48 {
    private static Evaluation evalCrossValidation;
    private static Evaluation evalPrecentageSplit;
    private static Evaluation evalTrainingTest;
    private static J48 j48;
    private static myC45 c45;
    private static Instances dataTraining;

    private static void printResult() throws Exception {
//        System.out.println(j48);
        System.out.println(c45);
        System.out.println("=============Cross Validation=============");
        System.out.println(evalCrossValidation.toSummaryString());
        System.out.println(evalCrossValidation.toClassDetailsString());
        System.out.println(evalCrossValidation.toMatrixString());

        System.out.println("=============Percentage Split=============");
        System.out.println(evalPrecentageSplit.toSummaryString());
        System.out.println(evalPrecentageSplit.toClassDetailsString());
        System.out.println(evalPrecentageSplit.toMatrixString());

        System.out.println("=============Training Test=============");
        System.out.println(evalTrainingTest.toSummaryString());
        System.out.println(evalTrainingTest.toClassDetailsString());
        System.out.println(evalTrainingTest.toMatrixString());
    }

    private static  void percentageSplit(double percent) throws Exception {
        int trainSize = (int) Math.round((dataTraining.numInstances()*percent)/100.0);
        int testSize = dataTraining.numInstances()-trainSize;

        Instances train =  new Instances (dataTraining,0, trainSize);
        Instances test = new Instances (dataTraining, trainSize, testSize);

//        J48 j48Percent = new J48();
//        train.setClassIndex(train.numAttributes() - 1);
//        j48Percent.buildClassifier(train);

        myC45 c45Percent = new myC45();
        train.setClassIndex(train.numAttributes() - 1);
        c45Percent.buildClassifier(train);


        evalPrecentageSplit = new Evaluation(test);
        evalPrecentageSplit.evaluateModel(c45Percent, test);

    }

    public static void saveModel(String model_name, J48 classifiers) throws Exception {
        weka.core.SerializationHelper.write(model_name, classifiers);
    }

    public static J48 loadModel(String model_name) throws Exception {
        J48 classifiers = (J48) weka.core.SerializationHelper.read(model_name);
        return classifiers;
    }

    public static void main(String[] args) throws java.lang.Exception {
        // load from arff

        ConverterUtils.DataSource source = new ConverterUtils.DataSource("data/weather.nominal.arff");
        Instances data = source.getDataSet();

        // remove attribut
        Remove rm = new Remove();
        rm.setAttributeIndices("1");  // remove 1st attribute

        // filter: resample
        Resample resample = new Resample();
        resample.setInputFormat(data);
        Instances dataResample = Filter.useFilter(data, resample);

        dataTraining = new Instances(dataResample);
        dataTraining.setClassIndex( dataTraining.numAttributes() - 1);

        // train j48
//        j48 = new J48();
//        dataResample.setClassIndex(dataResample.numAttributes() - 1);
//        j48.buildClassifier(dataResample);

        // train c45
        c45 = new myC45();
        dataResample.setClassIndex(dataResample.numAttributes() - 1);
        c45.buildClassifier(dataResample);


        // testing model given test set

        // testing
        evalTrainingTest = new Evaluation(dataResample);
        evalTrainingTest.evaluateModel(c45, dataResample);

        evalCrossValidation = new Evaluation(dataResample);
        evalCrossValidation.crossValidateModel(c45, dataResample, 10, new Random(1));

        percentageSplit(80.0);

        printResult();

        // input data

    }
}
