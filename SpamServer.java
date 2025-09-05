import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

/**
 * A full-stack Spam Classifier Backend Server.
 * This server loads and trains the Naive Bayes model once, then listens for
 * API requests to classify new email text.
 */
public class SpamServer {

    // FIX: The 'Email' record is now defined at the top level of the SpamServer class,
    // making it accessible to both the main method and the nested SpamClassifier class.
    private record Email(String message, String label) {}

    // The classifier instance that will hold the trained model.
    private static final SpamClassifier classifier = new SpamClassifier();

    public static void main(String[] args) throws IOException {
        // --- Step 1: Train the Model at Startup ---
        System.out.println("--- Initializing Spam Classifier ---");
        List<Email> allData = SpamClassifier.loadDataset("spam_ham_dataset.csv");
        if (allData.isEmpty()) {
            System.err.println("Dataset could not be loaded. Server cannot start.");
            return;
        }
        System.out.println("Successfully loaded " + allData.size() + " emails.");
        classifier.train(allData); // Train on the entire dataset
        System.out.println("--- Model training complete ---");

        // --- Step 2: Start the Web Server ---
        int port = 8080;
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        // Create an endpoint at "/classify" that will handle classification requests.
        server.createContext("/classify", new ClassificationHandler());
        server.setExecutor(null); // Use the default executor
        server.start();

        System.out.println("\n--- Server is running! ---");
        System.out.println("Listening on port: " + port);
        System.out.println("Send POST requests to http://localhost:8080/classify");
    }

    /**
     * This handler processes incoming requests to the /classify endpoint.
     */
    static class ClassificationHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            // --- CORS Headers: Allow the frontend to connect ---
            exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
            exchange.getResponseHeaders().add("Access-Control-Allow-Methods", "POST, OPTIONS");
            exchange.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");

            // Handle pre-flight CORS requests
            if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(204, -1); // No Content
                return;
            }

            if ("POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                // Read the email text from the request body
                InputStreamReader isr = new InputStreamReader(exchange.getRequestBody(), StandardCharsets.UTF_8);
                BufferedReader br = new BufferedReader(isr);
                String emailText = br.lines().collect(Collectors.joining("\n"));

                // Use the pre-trained classifier to predict
                String prediction = classifier.predict(emailText);

                // Create a JSON response
                String jsonResponse = "{\"classification\": \"" + prediction + "\"}";

                // Send the response back to the frontend
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, jsonResponse.getBytes().length);
                OutputStream os = exchange.getResponseBody();
                os.write(jsonResponse.getBytes());
                os.close();
            } else {
                // Handle invalid request methods
                exchange.sendResponseHeaders(405, -1); // Method Not Allowed
            }
        }
    }

    // --- The SpamClassifier class remains the same, but is now used by the server ---
    static class SpamClassifier {
        // The 'Email' record has been moved outside this class.

        private final Map<String, Integer> spamWordCounts = new HashMap<>();
        private final Map<String, Integer> hamWordCounts = new HashMap<>();
        private long spamEmailCount = 0;
        private long hamEmailCount = 0;
        private final Set<String> vocabulary = new HashSet<>();
        private double pSpam;
        private double pHam;

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

        public List<String> preprocessText(String text) {
            if (text == null) return new ArrayList<>();
            String cleanedText = text.toLowerCase().replaceAll("[^a-zA-Z0-9\\s]", "");
            return Arrays.stream(cleanedText.split("\\s+"))
                         .filter(word -> !word.isEmpty() && !STOP_WORDS.contains(word))
                         .collect(Collectors.toList());
        }

        public void train(List<Email> trainingData) {
            for (Email email : trainingData) {
                if ("spam".equalsIgnoreCase(email.label())) spamEmailCount++;
                else hamEmailCount++;
                
                List<String> words = preprocessText(email.message());
                for (String word : words) {
                    vocabulary.add(word);
                    Map<String, Integer> wordCounts = "spam".equalsIgnoreCase(email.label()) ? spamWordCounts : hamWordCounts;
                    wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
                }
            }

            long totalEmails = spamEmailCount + hamEmailCount;
            pSpam = (double) spamEmailCount / totalEmails;
            pHam = (double) hamEmailCount / totalEmails;
        }

        public String predict(String message) {
            List<String> words = preprocessText(message);
            double logProbSpam = Math.log(pSpam);
            double logProbHam = Math.log(pHam);
            long totalSpamWords = spamWordCounts.values().stream().mapToLong(Integer::intValue).sum();
            long totalHamWords = hamWordCounts.values().stream().mapToLong(Integer::intValue).sum();
            int vocabSize = vocabulary.size();

            for (String word : words) {
                double pWordSpam = Math.log((double) (spamWordCounts.getOrDefault(word, 0) + 1) / (totalSpamWords + vocabSize));
                logProbSpam += pWordSpam;
                double pWordHam = Math.log((double) (hamWordCounts.getOrDefault(word, 0) + 1) / (totalHamWords + vocabSize));
                logProbHam += pWordHam;
            }
            return logProbSpam > logProbHam ? "spam" : "ham";
        }

        public static List<Email> loadDataset(String filePath) {
            List<Email> dataset = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                br.readLine(); // Skip header
                String line;
                StringBuilder recordBuilder = new StringBuilder();
                while ((line = br.readLine()) != null) {
                    recordBuilder.append(line).append("\n");
                    long quoteCount = recordBuilder.chars().filter(ch -> ch == '\"').count();
                    if (quoteCount % 2 == 0) {
                        List<String> fields = parseCsvLine(recordBuilder.toString().trim());
                        if (fields.size() >= 3) {
                            dataset.add(new Email(fields.get(2), fields.get(1)));
                        }
                        recordBuilder.setLength(0);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            return dataset;
        }

        private static List<String> parseCsvLine(String line) {
            List<String> fields = new ArrayList<>();
            StringBuilder currentField = new StringBuilder();
            boolean inQuotes = false;
            for (int i = 0; i < line.length(); i++) {
                char c = line.charAt(i);
                if (c == '\"') {
                    if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '\"') {
                        currentField.append('\"'); i++;
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
    }
}
