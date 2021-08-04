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

package io.github.marcelovca90.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class Utils
{
    public static long run(String[] command, String outputFilename) throws IOException, InterruptedException
    {
        ProcessBuilder builder = new ProcessBuilder(command);

        if (outputFilename != null)
            builder = builder.redirectOutput(new File(outputFilename));

        long start = System.nanoTime();
        Process process = builder.start();
        process.waitFor();
        long end = System.nanoTime();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream())))
        {
            String line = "";
            while ((line = reader.readLine()) != null)
                System.out.println(line);
        }

        return (end - start) / 1000000L;
    }

    public static void balance(List<String> dataset, int seed)
    {
        Random random = new Random(seed);

        int hamCount = (int) dataset.stream().filter(i -> i.startsWith("1 ")).count();
        int spamCount = (int) dataset.stream().filter(i -> i.startsWith("2 ")).count();

        if (hamCount < spamCount)
            for (int i = 0; i < spamCount - hamCount; i++)
                dataset.add(dataset.get(random.nextInt(hamCount)));
        else if (spamCount < hamCount)
            for (int i = 0; i < hamCount - spamCount; i++)
                dataset.add(dataset.get(hamCount + random.nextInt(spamCount)));
    }

    public static void shuffle(List<String> dataset, int seed)
    {
        Random random = new Random(seed);

        int numberOfInstances = dataset.size();
        for (int i = 0; i < numberOfInstances; i++)
        {
            int j = random.nextInt(numberOfInstances);
            String a = dataset.get(i);
            String b = dataset.get(j);
            dataset.set(i, b);
            dataset.set(j, a);
        }
    }

    public static Pair<List<String>, List<String>> split(List<String> dataset, double splitPercent)
    {
        int numberOfInstances = dataset.size();

        List<String> trainSet = new ArrayList<>();
        for (int i = 0; i < (int) (splitPercent * numberOfInstances); i++)
            trainSet.add(dataset.get(i));

        List<String> testSet = new ArrayList<>();
        for (int i = (int) (splitPercent * numberOfInstances); i < numberOfInstances; i++)
            testSet.add(dataset.get(i));

        return Pair.of(trainSet, testSet);
    }

    public static void addEmpty(List<String> testSet, int emptyHamCount, int emptySpamCount)
    {
        for (int i = 0; i < emptyHamCount; i++)
            testSet.add("1");
        for (int i = 0; i < emptySpamCount; i++)
            testSet.add("2");
    }

    public static double confidenceInterval(DescriptiveStatistics statistics)
    {
        if (statistics.getN() <= 1)
            return 0.0;

        TDistribution tDist = new TDistribution(statistics.getN() - 1);
        double a = tDist.inverseCumulativeProbability(1.0 - 0.05 / 2);
        return a * statistics.getStandardDeviation() / Math.sqrt(statistics.getN());
    }

    public static void appendToFile(String filename, String value) throws IOException
    {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filename), true)))
        {
            writer.write(value);
            writer.flush();
        }
    }

    public static String formatPercentage(double v)
    {
        return String.format("%.2f", v);
    }

    public static String formatMillis(double millis)
    {
        return DurationFormatUtils.formatDurationHMS((Double.valueOf(Math.abs(millis))).longValue());
    }
}
