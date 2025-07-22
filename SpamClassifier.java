import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A complete, self-contained Spam Email Classifier using a Naive Bayes algorithm.
 * * How to use:
 * 1. Save this code as SpamClassifier.java.
 * 2. Download the "Spam Ham Dataset" from Kaggle (the file is named spam_ham_dataset.csv).
 * 3. Place 'spam_ham_dataset.csv' in the same directory as this Java file.
 * 4. Compile and run the program from your terminal:
 * javac SpamClassifier.java
 * java SpamClassifier
 */
public class SpamClassifier {

    // A simple record to hold email data
    private record Email(String message, String label) {}

    // --- Model Parameters ---
    private final Map<String, Integer> spamWordCounts = new HashMap<>();
    private final Map<String, Integer> hamWordCounts = new HashMap<>();
    private long spamEmailCount = 0;
    private long hamEmailCount = 0;
    private final Set<String> vocabulary = new HashSet<>();
    private double pSpam;
    private double pHam;

    // --- Preprocessing ---
    private static final Set<String> STOP_WORDS = new HashSet<>(Arrays.asList(
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
        "any", "are", "as", "at", "be", "because", "been", "before", "being", 
        "below", "between", "both", "but", "by", "can", "cannot", "could",
        "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", 
        "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", 
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", 
        "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", 
        "them", "themselves", "then", "there", "these", "they", "this", "those", "through", 
        "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", 
        "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", 
        "yours", "yourself", "yourselves"
    ));

    /**
     * Preprocesses a given text by cleaning and tokenizing it.
     */
    public List<String> preprocessText(String text) {
        if (text == null) {
            return new ArrayList<>();
        }
        String cleanedText = text.toLowerCase().replaceAll("[^a-zA-Z0-9\\s]", "");
        return Arrays.stream(cleanedText.split("\\s+"))
                     .filter(word -> !word.isEmpty() && !STOP_WORDS.contains(word))
                     .collect(Collectors.toList());
    }

    // --- Training ---
    /**
     * Trains the Naive Bayes model on a given dataset.
     */
    public void train(List<Email> trainingData) {
        System.out.println("Starting training...");

        for (Email email : trainingData) {
            if ("spam".equalsIgnoreCase(email.label())) {
                spamEmailCount++;
            } else {
                hamEmailCount++;
            }
            
            List<String> words = preprocessText(email.message());
            for (String word : words) {
                vocabulary.add(word);
                Map<String, Integer> wordCounts = "spam".equalsIgnoreCase(email.label()) ? spamWordCounts : hamWordCounts;
                wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
            }
        }

        long totalEmails = spamEmailCount + hamEmailCount;
        if (totalEmails > 0) {
            pSpam = (double) spamEmailCount / totalEmails;
            pHam = (double) hamEmailCount / totalEmails;
        } else {
            pSpam = 0.5;
            pHam = 0.5;
        }

        System.out.println("Training complete.");
        System.out.println("Total emails in training set: " + totalEmails);
        System.out.println("Spam emails: " + spamEmailCount + " | Ham emails: " + hamEmailCount);
        System.out.println("Vocabulary size: " + vocabulary.size());
    }

    // --- Prediction ---
    /**
     * Predicts whether a new email message is spam or ham.
     */
    public String predict(String message) {
        List<String> words = preprocessText(message);

        if (pSpam == 0 && pHam == 0) return "ham"; // Un-trained model
        if (pSpam == 0) return "ham";
        if (pHam == 0) return "spam";

        double logProbSpam = Math.log(pSpam);
        double logProbHam = Math.log(pHam);

        long totalSpamWords = spamWordCounts.values().stream().mapToLong(Integer::intValue).sum();
        long totalHamWords = hamWordCounts.values().stream().mapToLong(Integer::intValue).sum();
        int vocabSize = vocabulary.size();

        for (String word : words) {
            // Laplace (add-1) smoothing
            double pWordSpam = (double) (spamWordCounts.getOrDefault(word, 0) + 1) / (totalSpamWords + vocabSize);
            logProbSpam += Math.log(pWordSpam);

            double pWordHam = (double) (hamWordCounts.getOrDefault(word, 0) + 1) / (totalHamWords + vocabSize);
            logProbHam += Math.log(pWordHam);
        }

        return logProbSpam > logProbHam ? "spam" : "ham";
    }

    // --- Data Loading and Main Execution ---
    /**
     * Loads emails from the spam_ham_dataset.csv file.
     * This method uses a robust manual parser to handle commas, quotes, and newlines within the data fields.
     */
    public static List<Email> loadDataset(String filePath) {
        List<Email> dataset = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            br.readLine(); // Skip header line
            String line;
            StringBuilder recordBuilder = new StringBuilder();

            while ((line = br.readLine()) != null) {
                // If the record builder is empty, this is the start of a new potential record.
                // A valid new record should start with a number (the index).
                if (recordBuilder.length() == 0 && !line.matches("^\\d+.*")) {
                    continue; // Skip malformed lines that don't start with an index.
                }

                recordBuilder.append(line).append("\n"); // Use newline char for consistency

                // A complete record in this CSV file ends with ",0" or ",1" and has balanced quotes.
                // We check if the line seems to be a complete record.
                boolean looksComplete = line.endsWith(",0") || line.endsWith(",1");
                long quoteCount = recordBuilder.chars().filter(ch -> ch == '\"').count();

                if (looksComplete && quoteCount % 2 == 0) {
                    String record = recordBuilder.toString().trim();
                    List<String> fields = parseCsvLine(record);
                    
                    if (fields.size() >= 3) {
                        String label = fields.get(1);
                        String message = fields.get(2);
                        dataset.add(new Email(message, label));
                    }
                    // Reset for the next record
                    recordBuilder.setLength(0);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading dataset file: " + filePath);
            e.printStackTrace();
        }
        return dataset;
    }

    /**
     * A robust CSV line parser that handles quoted fields containing commas.
     * @param line A single line from a CSV file.
     * @return A list of fields from the line.
     */
    private static List<String> parseCsvLine(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder currentField = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '\"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '\"') {
                    currentField.append('\"');
                    i++; 
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (c == ',' && !inQuotes) {
                fields.add(currentField.toString());
                currentField.setLength(0);
            } else {
                currentField.append(c);
            }
        }
        fields.add(currentField.toString());
        return fields;
    }

    public static void main(String[] args) {
        System.out.println("--- Spam Email Classifier ---");
        
        List<Email> allData = loadDataset("spam_ham_dataset.csv");
        if (allData.isEmpty()) {
            System.out.println("Dataset could not be loaded. Please check the file path and format. Exiting.");
            return;
        }
        System.out.println("Successfully loaded " + allData.size() + " emails.");
        
        java.util.Collections.shuffle(allData);
        int splitIndex = (int) (allData.size() * 0.8);
        List<Email> trainingData = allData.subList(0, splitIndex);
        List<Email> testData = allData.subList(splitIndex, allData.size());

        SpamClassifier classifier = new SpamClassifier();
        classifier.train(trainingData);
        
        if (classifier.spamEmailCount == 0 || classifier.hamEmailCount == 0) {
            System.out.println("\nWarning: The model was trained on only one class of email. Accuracy will be poor.");
        }

        System.out.println("\n--- Testing Model ---");

        String testEmail1 = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!";
        String testEmail2 = "Hey, are we still on for the meeting tomorrow at 10am? Let me know.";
        String testEmail3 = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010";

        System.out.println("Email 1: \"" + testEmail1 + "\"");
        System.out.println("Prediction: " + classifier.predict(testEmail1) + "\n");
        
        System.out.println("Email 2: \"" + testEmail2 + "\"");
        System.out.println("Prediction: " + classifier.predict(testEmail2) + "\n");

        System.out.println("Email 3: \"" + testEmail3 + "\"");
        System.out.println("Prediction: " + classifier.predict(testEmail3) + "\n");

        int correctPredictions = 0;
        for (Email email : testData) {
            String prediction = classifier.predict(email.message());
            if (prediction.equalsIgnoreCase(email.label())) {
                correctPredictions++;
            }
        }

        double accuracy = 0.0;
        if (!testData.isEmpty()) {
            accuracy = (double) correctPredictions / testData.size() * 100;
        }
        
        System.out.println("--- Evaluation on Test Set ---");
        System.out.println("Total test emails: " + testData.size());
        System.out.println("Correct predictions: " + correctPredictions);
        System.out.printf("Accuracy: %.2f%%\n", accuracy);
    }
}
