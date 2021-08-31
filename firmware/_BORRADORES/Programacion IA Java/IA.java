/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.discord_bot_1;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.crypto.Mac;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.format.Mat5File;
import us.hebi.matlab.mat.types.Matrix;

/**
 *
 * @author Mentalink
 */
public class IA {

    final IUpdater UPDATER = Adam.builder().learningRate(0.001).beta1(0.5).build();
    final int Targets = 12;
    final int Channels = 8;
    final int Trials = 15;
    final int Largo_de_La_muestra = 1114;
    
    public static void main(String[] args) {
        try {
            IA ia = new IA();
        } catch (IOException ex) {
            Logger.getLogger(IA.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public IA() throws IOException {
        MultiLayerNetwork Network = Create_Network();
        DataSet[] ObtenerTestAndTrain = ObtenerTestAndTrain(true);
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 300; j++) {
                //6000 epocs en total
                Network.fit(ObtenerTestAndTrain[1]);
            }
            Evaluar(Network, ObtenerTestAndTrain[0]);
            ModelSerializer.writeModel(Network, "MentalinkIA.dl4j", true);
        }
    }
    
    public void Evaluar(MultiLayerNetwork Network,DataSet testData){
        Evaluation eval = new Evaluation(testData.getLabels().length());
        INDArray output = Network.output(testData.getFeatures());
        eval.eval(testData.getFeatures(), output);
        System.out.println(eval.stats());
    }
    
    public MultiLayerNetwork Create_Network() {
        MultiLayerConfiguration discConf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(UPDATER)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(100)
                .list()
                .layer(new DenseLayer.Builder().nIn(1114).nOut(2048).updater(UPDATER).build())
                .layer(new ActivationLayer.Builder(new ActivationReLU()).build())
                .layer(new DenseLayer.Builder().nIn(2048).nOut(1024).updater(UPDATER).build())
                .layer(new ActivationLayer.Builder(new ActivationReLU()).build())
                .layer(new DropoutLayer.Builder(1 - 0.5).build())
                .layer(new DenseLayer.Builder().nIn(1024).nOut(512).updater(UPDATER).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
                .layer(new DropoutLayer.Builder(1 - 0.5).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(256).updater(UPDATER).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.2)).build())
                .layer(new DropoutLayer.Builder(1 - 0.5).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(256).nOut(12)
                        .activation(Activation.SIGMOID).updater(UPDATER).build())
                .build();

        return new MultiLayerNetwork(discConf);
    }

    public DataSet[] ObtenerTestAndTrain(boolean Cargar_Datos) throws IOException {

        if (Cargar_Datos) {
            DataSet test = DataSet.empty();
            test.load(new File("DataTest.txt"));
            DataSet train = DataSet.empty();
            train.load(new File("DataTrain.txt"));
            return new DataSet[]{test,train};
        } else {
            DataSet Todos_Los_Datos = Cargar_Sujetos();
             SplitTestAndTrain testAndTrain = Todos_Los_Datos.splitTestAndTrain(0.9);
            DataSet dataset = testAndTrain.getTrain();
            DataSet testdata = testAndTrain.getTest();
            boolean fitlabels = false;
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(dataset);
            normalizer.fitLabel(fitlabels);//Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            normalizer.transform(dataset);     //Apply normalization to the training data
            normalizer.transform(testdata);
            testdata.save(new File("DataTest.txt"));
            dataset.save(new File("DataTrain.txt"));
            return new DataSet[]{testdata,dataset};
        }
        
    }

    public DataSet Cargar_Sujetos() throws IOException {

        List<List<Writable>> Data = new ArrayList<>();

        for (int sujeto = 0; sujeto < 11; sujeto++) {

            //<editor-fold desc="          Analizar existencia          ">
            if (sujeto == 8) {
                continue;
            }

            if (!new File("s" + sujeto + ".mat").exists()) {
                System.out.println("No se cargo [" + ("s" + sujeto + ".mat") + "]: El archivo no existe");
                continue;
            }
            //</editor-fold>

            //<editor-fold desc="          Obtener la matrix del Sujeto          ">
            Mat5File mat5File = Mat5.readFromFile("s" + sujeto + ".mat");

            Matrix matrix = null;

            try {
                matrix = mat5File.getMatrix("train");
            } catch (Exception e) {
                //Usualmente sale un error de classCastException si no es una matrix y vos pedis una ¿
                System.out.println("No se pudo obtener la matrix: [" + ("s" + sujeto + ".mat") + "] "
                        + "no es una matrix de datos y/o no pertenece a los datasets de la competencia");
                continue;
            }
            //</editor-fold>

            //<editor-fold desc="          Pasar datos de Matrix --> List<List<Writable>>>          ">
            for (int trg = 0; trg < Targets; trg++) {
                for (int channel = 0; channel < Channels; channel++) {
                    for (int trial = 0; trial < Trials; trial++) {
                        List<Writable> _Data = new ArrayList<>();
                        try {

                            double minval = 100000;
                            for (int i = 0; i < Largo_de_La_muestra; i++) {
                                try {
                                    if (i == 0) {
                                        minval = matrix.getDouble(new int[]{trg, channel, i, trial});
                                    } else if (matrix.getDouble(new int[]{trg, channel, i, trial}) < minval) {
                                        minval = matrix.getDouble(new int[]{trg, channel, i, trial});
                                    }
                                } catch (Exception e) {
                                    //proteccion contra IndexOutOfBounds
                                    break;
                                }
                            }
                            for (int i = 0; i < Largo_de_La_muestra; i++) {
                                try {
                                    //Le restamos MinValue para mantener todos los datos en el mismo rango al rededor de 0
                                    //No se si esta bien xd
                                    _Data.add(new DoubleWritable(matrix.getDouble(new int[]{trg, channel, i, trial}) - minval));
                                } catch (Exception e) {
                                    //proteccion contra IndexOutOfBounds
                                    break;
                                }

                            }
                            //Añadimos al final del arreglo el target al que pertenece
                            _Data.add(new DoubleWritable(trg));

                        } catch (Exception e) {
                            //proteccion contra IndexOutOfBounds
                            continue;
                        }

                        Data.add(_Data);
                    }

                }
            }
            //</editor-fold>

        }

        //<editor-fold desc="          Pasar datos de List<List<Writable>>> --> DataSet         ">
        RecordReader rr = new CollectionRecordReader(Data);

        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(rr, Data.size(), 1114, 12);

        DataSetIterator iterator = iter;

        DataSet alldata = null;
        try {
            alldata = iterator.next();
        } catch (Exception e) {
            System.out.println("Un error critico a ocurrido con los datasets\nNo se pudo obtener el Dataset: Las dependencias estan mal añadidas o faltan dependencias en el POM");

            e.printStackTrace();
            System.exit(0);
        }
        alldata.shuffle();
        //</editor-fold>

        return alldata;
    }

}
