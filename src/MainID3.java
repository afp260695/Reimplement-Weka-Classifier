import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Resample;

import java.util.Random;

public class MainID3 {
    private static Evaluation evalCrossValidation;
    private static Evaluation evalPrecentageSplit;
    private static Evaluation evalTrainingTest;
    private static Id3 id3;
    private static Instances dataTraining;

    private static void printResult() throws Exception {
        System.out.println(id3);
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

        Id3 id3Percent = new Id3();
        train.setClassIndex(train.numAttributes() - 1);
        id3Percent.buildClassifier(train);

        evalPrecentageSplit = new Evaluation(test);
        evalPrecentageSplit.evaluateModel(id3Percent, test);

    }

    public static void saveModel(String model_name, Id3 classifiers) throws Exception {
        weka.core.SerializationHelper.write(model_name, classifiers);
    }

    public static Id3 loadModel(String model_name) throws Exception {
        Id3 classifiers = (Id3) weka.core.SerializationHelper.read(model_name);
        return classifiers;
    }

    public static void main(String[] args) throws java.lang.Exception {
        // load from arff

        DataSource source = new DataSource("data/iris.2D.arff");
        Instances data = source.getDataSet();

        // remove attribut
        Remove rm = new Remove();
        rm.setAttributeIndices("1");  // remove 1st attribute

        // filter: resample
        Resample resample = new Resample();
        resample.setInputFormat(data);
        Instances dataResample = Filter.useFilter(data, resample);

        Discretize discretize = new Discretize();
        discretize.setInputFormat(dataResample);
        Instances dataDiscritize = Filter.useFilter(dataResample, discretize);
        dataTraining = new Instances(dataDiscritize);
        dataTraining.setClassIndex( dataTraining.numAttributes() - 1);

        // train id3
        id3 = new Id3();
        dataDiscritize.setClassIndex(dataDiscritize.numAttributes() - 1);
        id3.buildClassifier(dataDiscritize);

        // testing model given test set

        // testing
        evalTrainingTest = new Evaluation(dataDiscritize);
        evalTrainingTest.evaluateModel(id3, dataDiscritize);

        evalCrossValidation = new Evaluation(dataDiscritize);
        evalCrossValidation.crossValidateModel(id3, dataDiscritize, 10, new Random(1));

        percentageSplit(80.0);

        printResult();

        // input data

    }
}
