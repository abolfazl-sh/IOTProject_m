/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;

/**
 *
 * @author AAA
 */
public class MyKafkaProducer{
    
    private final static String TOPIC = "IOT-data";
    private final static String TRAIN_TOPIC = "IOT-train";
    private final static String TEST_TOPIC = "IOT-test";
    private final static String BOOTSTRAP_SERVERS = "172.16.2.200:9092";
    
    Path path = FileSystems.getDefault().getPath("").toAbsolutePath();
    
    private static Producer<String, String> createProducer(){
        
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, BOOTSTRAP_SERVERS);
        props.put(ProducerConfig.CLIENT_ID_CONFIG, "KafkaIOTProducer1");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        
        KafkaProducer a = new KafkaProducer<>(props);
        return new KafkaProducer<>(props);
    }
    
    final static Producer<String, String> producer = createProducer();
    
    static void runProducer(String topic,String key,String value) throws Exception{
        
        long time = System.currentTimeMillis();
        try{
            
                final ProducerRecord<String, String> record =
                        new ProducerRecord<>(topic, key, value);
                
                producer.send(record).get();

            
        }finally{
            producer.flush();
            
        }
    }
  
    
    public static void main(String[] args) throws Exception{
        System.out.println("hi");
        org.apache.log4j.BasicConfigurator.configure();
        Path path = FileSystems.getDefault().getPath("").toAbsolutePath();
        BufferedReader trainreader = new BufferedReader(new FileReader(path + "\\Data\\train.csv"));
        BufferedReader testreader = new BufferedReader(new FileReader(path + "\\Data\\test.csv"));
        String[] headertrain = trainreader.readLine().split(",");
        String[] headertest = testreader.readLine().split(",");
        

        
            Thread test = new Thread(new Runnable() {
            @Override
            public void run() {
                int testIndex = 0;
                try {
                    while(testreader.ready()){
                        
                            runProducer(TEST_TOPIC, testIndex + "", testreader.readLine());
                            System.out.println("test : " + testreader.readLine());
                            Thread.sleep(1000);
                            testIndex++;
                        
                    }
                } catch (IOException ex) {
                    Logger.getLogger(MyKafkaProducer.class.getName()).log(Level.SEVERE, null, ex);
                } catch (Exception ex) {
                    Logger.getLogger(MyKafkaProducer.class.getName()).log(Level.SEVERE, null, ex);
                }
                 
            }
        });
        
        Thread train = new Thread(new Runnable() {
            @Override
            public void run() {
                int trainIndex = 0;
                try {
                    while(trainreader.ready()){
                        
                            runProducer(TRAIN_TOPIC, trainIndex + "" , trainreader.readLine());
                            Thread.sleep(10000);
                            
                     
                    }
                } catch (IOException ex) {
                    Logger.getLogger(MyKafkaProducer.class.getName()).log(Level.SEVERE, null, ex);
                } catch (InterruptedException ex) {
                    Logger.getLogger(MyKafkaProducer.class.getName()).log(Level.SEVERE, null, ex);
                } catch (Exception ex) {
                    Logger.getLogger(MyKafkaProducer.class.getName()).log(Level.SEVERE, null, ex);
                }
                test.start();
            }
        });
        

        
        train.start();
       
        
        
     /*   org.apache.log4j.BasicConfigurator.configure();
        int count = 0;
        while(true){
        Thread.sleep(100);
        runProducer(1);    
        count++;
        if(count > 100){
            break;
        }
        }
        
        producer.close();*/
        
    }
}
