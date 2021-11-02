/*******************************************************************************
 * Copyright (C) 2017 Marcelo Vinícius Cysneiros Aragão
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

package io.github.marcelovca90.main;

import static io.github.marcelovca90.util.Utils.addEmpty;
import static io.github.marcelovca90.util.Utils.appendToFile;
import static io.github.marcelovca90.util.Utils.balance;
import static io.github.marcelovca90.util.Utils.confidenceInterval;
import static io.github.marcelovca90.util.Utils.formatMillis;
import static io.github.marcelovca90.util.Utils.formatPercentage;
import static io.github.marcelovca90.util.Utils.run;
import static io.github.marcelovca90.util.Utils.shuffle;
import static io.github.marcelovca90.util.Utils.split;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.github.habernal.confusionmatrix.ConfusionMatrix;

import io.github.marcelovca90.math.Integration;

public class Runner
{
    private static final String USER_HOME = System.getProperty("user.home");
    private static final String REPO_FOLDER = "/git/marcelovca90-unifei/liblinear-marcelovca90";

    public static void main(String[] args) throws Exception
    {
        if (args == null || args.length == 0)
            args = new String[] { "empty" };

        switch (args[0])
        {
            case "prepare":
                prepare(args);
                break;

            case "train":
                train(args);
                break;

            case "test":
                test(args);
                break;

            case "evaluate":
                evaluate(args);
                break;

            case "aggregate":
                aggregate(args);
                break;

            default:
                throw new Exception("usage: java -jar arff2liblinear.jar prepare|train|test|evaluate|aggregate");
        }
    }

    private static void prepare(String[] args) throws Exception, IOException
    {
        if (args.length != 5)
            throw new Exception(
                "usage: java -jar arff2liblinear.jar prepare arff_filename empty_ham_count empty_spam_count seed");

        String arffFilename = args[1];
        int emptyHamCount = Integer.parseInt(args[2]);
        int emptySpamCount = Integer.parseInt(args[3]);
        int seed = Integer.parseInt(args[4]);

        String output = arffFilename.replace("data.arff", "data.liblinear");

        StringBuilder sb = new StringBuilder();

        List<String> dataset = new ArrayList<>();

        Files.readAllLines(Paths.get(arffFilename)).stream().forEach(line ->
        {
            int y = line.endsWith("HAM") ? 1 : line.endsWith("SPAM") ? 2 : Integer.MIN_VALUE;

            if (y != Integer.MIN_VALUE)
            {
                String[] parts = line.split(",");

                sb.append(y);
                for (int i = 0; i < parts.length - 1; i++)
                    sb.append(String.format(" %d:%s", i + 1, parts[i]));

                dataset.add(sb.toString());
                sb.setLength(0);
            }
        });

        balance(dataset, seed);
        shuffle(dataset, seed);
        FileUtils.writeLines(new File(output), dataset);

        Pair<List<String>, List<String>> datasets = split(dataset, 0.5);

        List<String> trainSet = datasets.getLeft();
        FileUtils.writeLines(new File(output.replace("data.liblinear", "data.train")), trainSet);

        List<String> testSet = datasets.getRight();
        addEmpty(testSet, emptyHamCount, emptySpamCount);
        FileUtils.writeLines(new File(output.replace("data.liblinear", "data.test")), testSet);
    }

    private static void train(String[] args) throws Exception
    {
        if (args.length < 3 || (!args[1].equals("poly1") && !args[1].equals("poly2")))
            throw new Exception(
                "usage: java -jar arff2liblinear.jar train poly1|poly2 [solver0|1|2|3|4|5|6|7] training_set_file");

        String exec = args[1].equals("poly1") ? "liblinear-default" : "liblinear-poly2-2.01";
        String solver = args[2].matches("solver[0-7]") ? args[2].substring(args[2].length() - 1) : "1";
        String training_set_file = args[3];
        String model_file = training_set_file.replace(".train", ".model");
        String best_c_file = training_set_file.replace(".train", ".best_c");
        boolean find_best_c = false;
        double best_c = 128.0;
        String[] cmd;

        // find best C
        if (find_best_c && (solver.equals("0") || solver.equals("2")) && !Files.exists(Paths.get(best_c_file)))
        {
            cmd = new String[] {
                    USER_HOME + "/git/liblinear-marcelovca90/" + exec + "/train",
                    "-s",
                    solver,
                    "-C",
                    training_set_file,
                    model_file
            };
            run(cmd, best_c_file);
            Optional<String> optional = Files.lines(Paths.get(best_c_file)).filter(l -> l.startsWith("Best C")).findFirst();
            best_c = optional.isPresent() ? Double.parseDouble(optional.get().split("\\s")[3]) : best_c;
        }

        // train
        cmd = new String[] {
                USER_HOME + REPO_FOLDER + "/" + exec + "/train",
                "-s",
                solver,
                "-c",
                String.valueOf(best_c),
                "-q",
                training_set_file,
                model_file
        };
        long execTime = run(cmd, null);

        String execTimeFilename = training_set_file.replace(".train", ".train_times");
        appendToFile(execTimeFilename, String.valueOf(execTime) + "\n");
    }

    private static void test(String[] args) throws Exception
    {
        if (args.length != 4 || (args.length == 4 && !args[1].equals("poly1") && !args[1].equals("poly2")))
            throw new Exception("usage: java -jar arff2liblinear.jar test poly1|poly2 test_file model_file");

        String exec = args[1].equals("poly1") ? "liblinear-default" : "liblinear-poly2-2.01";
        String test_file = args[2];
        String model_file = args[3];
        String output_file = test_file.replace(".test", ".prediction");

        String[] cmd = new String[] {
                USER_HOME + REPO_FOLDER + "/" + exec + "/predict",
                test_file,
                model_file, output_file
        };

        long execTime = run(cmd, null);

        String execTimeFilename = test_file.replace(".test", ".test_times");
        appendToFile(execTimeFilename, String.valueOf(execTime) + "\n");
    }

    private static void evaluate(String[] args) throws Exception
    {
        if (args.length != 3)
            throw new Exception("usage: java -jar arff2liblinear.jar evaluate test_filename prediction_filename");

        String testFilename = args[1];
        String predictionFilename = args[2];

        List<Integer> expected = Files
            .readAllLines(Paths.get(testFilename))
            .stream()
            .map(line -> Character.getNumericValue(line.charAt(0)))
            .collect(Collectors.toList());

        List<Integer> predicted = Files
            .readAllLines(Paths.get(predictionFilename))
            .stream()
            .map(line -> Character.getNumericValue(line.charAt(0)))
            .collect(Collectors.toList());

        if (expected.size() != predicted.size())
            throw new Exception("expected.size() != predicted.size()");

        int fp = 0, tn = 0, fn = 0, tp = 0;

        for (int i = 0; i < expected.size(); i++)
        {
            if (expected.get(i) == 1 && predicted.get(i) == 1)
                tn++; // TN ham_ham
            if (expected.get(i) == 1 && predicted.get(i) == 2)
                fp++; // FP ham_spam
            if (expected.get(i) == 2 && predicted.get(i) == 1)
                fn++; // FN spam_ham
            if (expected.get(i) == 2 && predicted.get(i) == 2)
                tp++; // TP spam_spam
        }

        ConfusionMatrix cm = new ConfusionMatrix();

        cm.increaseValue("ham", "ham", tn);
        cm.increaseValue("ham", "spam", fp);
        cm.increaseValue("spam", "ham", fn);
        cm.increaseValue("spam", "spam", tp);

        double hamPrecision = 100.0 * cm.getPrecisionForLabel("ham");
        double spamPrecision = 100.0 * cm.getPrecisionForLabel("spam");
        double avgPrecision = 100.0 * cm.getAvgPrecision();
        double hamRecall = 100.0 * cm.getRecallForLabel("ham");
        double spamRecall = 100.0 * cm.getRecallForLabel("spam");
        double avgRecall = 100.0 * cm.getAvgRecall();
        double hamfMeasure = 100.0 * cm.getFMeasureForLabels().get("ham");
        double spamfMeasure = 100.0 * cm.getFMeasureForLabels().get("ham");
        double fMeasure = 100.0 * (2.0 * (1.0 / ((1.0 / cm.getAvgRecall()) + (1.0 / cm.getAvgPrecision()))));

        // https://en.wikipedia.org/wiki/Precision_and_recall
        // https://stats.stackexchange.com/questions/7207/roc-vs-precision-and-recall-curves

        double x_recall = ((double) tp) / ((double) (tp + fn));
        double y_precision = ((double) tp) / ((double) (tp + fp));
        double[] xPRC = { 0.0, x_recall, 1.0 };
        double[] yPRC = { 0.0, y_precision, 1.0 };
        double auPRC = 100.0 * Integration.trapz(xPRC, yPRC);

        double x_fpr = ((double) fp) / ((double) (fp + tn));
        double y_tpr = ((double) tp) / ((double) (tp + fn));
        double[] xROC = { 0.0, x_fpr, 1.0 };
        double[] yROC = { 0.0, y_tpr, 1.0 };
        double auROC = 100.0 * Integration.trapz(xROC, yROC);

        String result = String
            .format(
                "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
                hamPrecision, spamPrecision, avgPrecision, hamRecall, spamRecall, avgRecall,
                auPRC, auROC, hamfMeasure, spamfMeasure, fMeasure);

        String outputFilename = predictionFilename.replace(".prediction", ".partial_results");
        appendToFile(outputFilename, result);
    }

    private static void aggregate(String[] args) throws Exception
    {
        if (args.length != 4)
            throw new Exception("usage: java -jar arff2liblinear.jar aggregate partial_results train_times test_times");

        Map<String, DescriptiveStatistics> map = new HashMap<>();
        map.put("hamPrecision", new DescriptiveStatistics());
        map.put("spamPrecision", new DescriptiveStatistics());
        map.put("avgPrecision", new DescriptiveStatistics());
        map.put("hamRecall", new DescriptiveStatistics());
        map.put("spamRecall", new DescriptiveStatistics());
        map.put("avgRecall", new DescriptiveStatistics());
        map.put("areaUnderPRC", new DescriptiveStatistics());
        map.put("areaUnderROC", new DescriptiveStatistics());
        map.put("hamFMeasure", new DescriptiveStatistics());
        map.put("spamFMeasure", new DescriptiveStatistics());
        map.put("fMeasure", new DescriptiveStatistics());
        map.put("trainTime", new DescriptiveStatistics());
        map.put("testTime", new DescriptiveStatistics());

        String partialResultsFilename = args[1];
        String trainTimesFilename = args[2];
        String testTimesFilename = args[3];

        Files.readAllLines(Paths.get(partialResultsFilename)).stream().forEach(line ->
        {
            String[] parts = Arrays.stream(line.split("\t", -1)).map(i -> i.replace(',', '.')).toArray(String[]::new);
            map.get("hamPrecision").addValue(Double.parseDouble(parts[0]));
            map.get("spamPrecision").addValue(Double.parseDouble(parts[1]));
            map.get("avgPrecision").addValue(Double.parseDouble(parts[2]));
            map.get("hamRecall").addValue(Double.parseDouble(parts[3]));
            map.get("spamRecall").addValue(Double.parseDouble(parts[4]));
            map.get("avgRecall").addValue(Double.parseDouble(parts[5]));
            map.get("areaUnderPRC").addValue(Double.parseDouble(parts[6]));
            map.get("areaUnderROC").addValue(Double.parseDouble(parts[7]));
            map.get("hamFMeasure").addValue(Double.parseDouble(parts[8]));
            map.get("spamFMeasure").addValue(Double.parseDouble(parts[9]));
            map.get("fMeasure").addValue(Double.parseDouble(parts[10]));
        });

        Files
            .readAllLines(Paths.get(trainTimesFilename)).stream()
            .forEach(line -> map.get("trainTime").addValue(Long.parseLong(line)));

        Files
            .readAllLines(Paths.get(testTimesFilename)).stream()
            .forEach(line -> map.get("testTime").addValue(Long.parseLong(line)));

        String aswd = "anti-spam-weka-data";
        String shortFilename = partialResultsFilename
            .substring(partialResultsFilename.indexOf(aswd) + aswd.length() + 1);
        shortFilename = shortFilename.substring(0, shortFilename.lastIndexOf(File.separator));
        shortFilename = Arrays.stream(shortFilename.split(File.separator)).collect(Collectors.joining("\t"));

        System.out.println(
            String.format(
                "%s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s\t%s ± %s",
                shortFilename, 
                formatPercentage(map.get("hamPrecision").getMean()), formatPercentage(confidenceInterval(map.get("hamPrecision"))),
                formatPercentage(map.get("spamPrecision").getMean()), formatPercentage(confidenceInterval(map.get("spamPrecision"))),
                formatPercentage(map.get("avgPrecision").getMean()), formatPercentage(confidenceInterval(map.get("avgPrecision"))),
                formatPercentage(map.get("hamRecall").getMean()), formatPercentage(confidenceInterval(map.get("hamRecall"))),
                formatPercentage(map.get("spamRecall").getMean()), formatPercentage(confidenceInterval(map.get("spamRecall"))),
                formatPercentage(map.get("avgRecall").getMean()), formatPercentage(confidenceInterval(map.get("avgRecall"))),
                formatPercentage(map.get("areaUnderPRC").getMean()), formatPercentage(confidenceInterval(map.get("areaUnderPRC"))),
                formatPercentage(map.get("areaUnderROC").getMean()), formatPercentage(confidenceInterval(map.get("areaUnderROC"))),
                formatPercentage(map.get("hamFMeasure").getMean()), formatPercentage(confidenceInterval(map.get("hamFMeasure"))),
                formatPercentage(map.get("spamFMeasure").getMean()), formatPercentage(confidenceInterval(map.get("spamFMeasure"))),
                formatPercentage(map.get("fMeasure").getMean()), formatPercentage(confidenceInterval(map.get("fMeasure"))), 
                formatMillis(map.get("trainTime").getMean()), formatMillis(confidenceInterval(map.get("trainTime"))),
                formatMillis(map.get("testTime").getMean()), formatMillis(confidenceInterval(map.get("testTime")))));
    }
}
