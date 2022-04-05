/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;


import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import org.apache.kafka.clients.consumer.ConsumerConfig;

import org.apache.kafka.clients.consumer.ConsumerRecord;

import org.apache.kafka.common.serialization.LongDeserializer;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.StreamingLogisticRegressionWithSGD;

import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import org.apache.spark.streaming.dstream.DStream;
import org.apache.spark.streaming.kafka010.ConsumerStrategies;
import org.apache.spark.streaming.kafka010.KafkaUtils;
import org.apache.spark.streaming.kafka010.LocationStrategies;



import scala.Tuple2;
/**
 *
 * @author AAA
 */
public class SparkStreamer {
    
    static Map<String, Object> kafkaParams = new HashMap<>();
    private final static String TOPIC = "IOT-data";
    private final static String BOOTSTRAP_SERVERS = "172.16.2.200:9092";

    public static void main(String[] args)  {
        // Create a local StreamingContext with two working thread and batch interval of 1 second
        SparkConf conf = new SparkConf().setMaster("yarn").setAppName("Sampleapp");
        JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(10));
        kafkaParams.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, BOOTSTRAP_SERVERS);
        kafkaParams.put(ConsumerConfig.GROUP_ID_CONFIG, "1");
        kafkaParams.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, LongDeserializer.class);
        kafkaParams.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        kafkaParams.put("auto.offset.reset", "earliest"); // from-beginning?
        kafkaParams.put("enable.auto.commit", false);

        Collection<String> topics = Arrays.asList(TOPIC);
        
        final JavaDStream<ConsumerRecord<Long, String>> stream =
                KafkaUtils.createDirectStream(
                        jssc,
                        LocationStrategies.PreferConsistent(),
                        ConsumerStrategies.<Long, String>Subscribe(topics, kafkaParams)
                );
        
        final JavaDStream<Vector> stream1 =
                KafkaUtils.createDirectStream(
                        jssc,
                        LocationStrategies.PreferConsistent(),
                        ConsumerStrategies.<Long, String>Subscribe(topics, kafkaParams)
                );
    
        
        StreamingLogisticRegressionWithSGD model1 = new StreamingLogisticRegressionWithSGD();
        model1.predictOn(stream);
        
        System.out.println("Direct Stream created? ");
        
        stream.foreachRDD(rdd -> System.out.printf("Amount of XMLs: %d\n", rdd.count()));

        jssc.start();
    try {
      jssc.awaitTermination();
    } catch (InterruptedException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
        System.out.println("Reached the end.");
    }
        
}
    

