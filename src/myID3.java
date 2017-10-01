import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Enumeration;

public class myID3 extends Classifier {
    private myID3[] m_Successors;
    private Attribute m_Attribute;
    private double m_ClassValue;
    private double[] m_Distribution;
    private Attribute m_ClassAttribute;

    public myID3(){

    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }

    public void buildClassifier(Instances data) throws Exception {
        this.getCapabilities().testWithFail(data);
        data = new Instances(data);
        data.deleteWithMissingClass();
        this.makeTree(data);
    }

    private void makeTree(Instances data) throws Exception {
        if (data.numInstances() == 0) {
            this.m_Attribute = null;
            this.m_ClassValue = Instance.missingValue();
            this.m_Distribution = new double[data.numClasses()];
        } else {

            double entropyValue = computeEntropy(data);
            // Mengecek apakah semua berada dalam satu kelas
            if(Utils.eq(entropyValue,0.0D)) {
                this.m_Attribute = null;
                this.m_Distribution = new double[data.numClasses()];

                Instance inst = data.firstInstance();
                this.m_Distribution[(int)inst.classValue()]++;

                this.m_ClassValue = (double)Utils.maxIndex(this.m_Distribution);
                this.m_ClassAttribute = data.classAttribute();

                // Mengecek apakah atrribut nya tinggal kelas saja
            } else if(data.numAttributes() == 1) {
                this.m_Attribute = null;
                this.m_Distribution = new double[data.numClasses()];

                Instance inst;
                for(Enumeration instEnum = data.enumerateInstances(); instEnum.hasMoreElements(); ++this.m_Distribution[(int)inst.classValue()]) {
                    inst = (Instance)instEnum.nextElement();
                }

                Utils.normalize(this.m_Distribution);
                this.m_ClassValue = (double)Utils.maxIndex(this.m_Distribution);
                this.m_ClassAttribute = data.classAttribute();
            } else {
                double[] infoGains = new double[data.numAttributes()];

                Attribute att;

                // Menghitung information gain tiap attribut
                for(Enumeration attEnum = data.enumerateAttributes(); attEnum.hasMoreElements(); infoGains[att.index()] = this.computeInfoGain(data, att)) {
                    att = (Attribute)attEnum.nextElement();
                }

                this.m_Attribute = data.attribute(Utils.maxIndex(infoGains));
                Instances[] splitData = this.splitData(data, this.m_Attribute);
                this.m_Successors = new myID3[this.m_Attribute.numValues()];
                Remove rm = new Remove();
                Integer indexAttribute = Utils.maxIndex(infoGains) + 1;
                rm.setAttributeIndices(indexAttribute.toString());  // remove 1st attribute

                for(int j = 0; j < this.m_Attribute.numValues(); ++j) {
                    rm.setInputFormat(splitData[j]);
                    // Menghapus attribut yang memiliki information gain yang paling besar, agar tidak dihitung di iterasi selanjutnya
                    Instances instances = Filter.useFilter(splitData[j],rm);
                    this.m_Successors[j] = new myID3();
                    this.m_Successors[j].makeTree(instances);
                }
            }

        }

    }

    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];

        for(int j = 0; j < att.numValues(); ++j) {
            splitData[j] = new Instances(data, data.numInstances());
        }

        Enumeration instEnum = data.enumerateInstances();

        while(instEnum.hasMoreElements()) {
            Instance inst = (Instance)instEnum.nextElement();
            splitData[(int)inst.value(att)].add(inst);
        }

        for(int i = 0; i < splitData.length; ++i) {
            splitData[i].compactify();
        }

        return splitData;
    }

    private double computeInfoGain(Instances data, Attribute att) throws Exception {
        double infoGain = this.computeEntropy(data);
        Instances[] splitData = this.splitData(data, att);

        for(int j = 0; j < att.numValues(); ++j) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= (double)splitData[j].numInstances() / (double)data.numInstances() * this.computeEntropy(splitData[j]);
            }
        }

        return infoGain;
    }

    private double computeEntropy(Instances data) throws Exception {
        double[] classCounts = new double[data.numClasses()];

        Instance inst;
        for(Enumeration instEnum = data.enumerateInstances(); instEnum.hasMoreElements(); ++classCounts[(int)inst.classValue()]) {
            inst = (Instance)instEnum.nextElement();
        }

        double entropy = 0.0D;

        for(int j = 0; j < data.numClasses(); ++j) {
            if (classCounts[j] > 0.0D) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }

        entropy /= (double)data.numInstances();
        return entropy + Utils.log2((double)data.numInstances());
    }

    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, please.");
        } else {
            return this.m_Attribute == null ? this.m_ClassValue : this.m_Successors[(int)instance.value(this.m_Attribute)].classifyInstance(instance);
        }
    }

    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, please.");
        } else {
            return this.m_Attribute == null ? this.m_Distribution : this.m_Successors[(int)instance.value(this.m_Attribute)].distributionForInstance(instance);
        }
    }

    private String toString(int level) {
        StringBuffer text = new StringBuffer();
        if (this.m_Attribute == null) {
            if (Instance.isMissingValue(this.m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": " + this.m_ClassAttribute.value((int)this.m_ClassValue));
            }
        } else {
            for(int j = 0; j < this.m_Attribute.numValues(); ++j) {
                text.append("\n");

                for(int i = 0; i < level; ++i) {
                    text.append("|  ");
                }

                text.append(this.m_Attribute.name() + " = " + this.m_Attribute.value(j));
                text.append(this.m_Successors[j].toString(level + 1));
            }
        }

        return text.toString();
    }

    public static void main(String[] args) {
        runClassifier(new myID3(), args);
    }
}
