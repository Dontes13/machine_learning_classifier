#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <sstream>
#include "csvstream.hpp"
#include <map>
#include <set>

using namespace std;



class Classifier {
  public:
    void train(const string& filename);

    string predict(const vector<string>& post) const;

    void accuracy(const string& filename);

    void print_data(const string& filename);

    double calculate_log_prob(const vector<string>& sorted_post, 
                              const string& label) const;

    set<string> unique_words(const string &str) {
      istringstream source(str);
      set<string> words;
      string word;
      while (source >> word) {
        words.insert(word);
      }
      return words;
    }

  private:
    int total_posts;
    int vocab;
    map<string, int> posts_per_word;
    set<string> all_words;
    set<string> labels;
    map<string, int> posts_per_label;
    map<string, map<string, int>> word_count_per_label;
};


void Classifier::train(const string& filename){
  total_posts = 0;
  vocab = 0;

  csvstream file(filename);
  map<string, string> line;

  while (file >> line){
    string label = line["tag"];
    string content = line["content"];

    labels.insert(label);
    total_posts++;
    posts_per_label[label]++;

    set<string> words = unique_words(content);
    for (auto it = words.begin(); it != words.end(); it++){
      all_words.insert(*it);
      posts_per_word[*it]++;
      word_count_per_label[label][*it]++;
    }
  }
  vocab = all_words.size();

}

string Classifier::predict(const vector<string>& post) const {
  // Sort only unique words
  set<string> unique;
  for (int i = 0; i < post.size(); i++){
    unique.insert(post[i]);
  }
  vector<string> sorted_post;
  for (auto it = unique.begin(); it != unique.end(); it++){
    sorted_post.push_back(*it);
  }
  double best_score;
  string best_label;

  for (auto it = labels.begin(); it != labels.end(); it++){
    string label = *it;
    double score = calculate_log_prob(sorted_post, label);

    // Set first score calculated to then calculate max score
    if (it == labels.begin()){
      best_score = score;
      best_label = label;
    }
    
    if (score > best_score){
      best_score = score;
      best_label = label;
    }

  }

  return best_label;
  
}


void Classifier::accuracy(const string& filename) {
  int correct_predictions = 0;
  int total_predictions = 0;

  cout << "trained on " << total_posts << " examples" << "\n" << endl;

  cout << "test data:" << endl;
  csvstream file(filename);
  map<string, string> line;

  while (file >> line){
    string content = line["content"];
    string correct_label = line["tag"];
    set<string> unique = unique_words(content);
    vector<string> post;
    for (auto it = unique.begin(); it != unique.end(); it++){
      post.push_back(*it);
    }
    string predicted = predict(post);
    double log_score = calculate_log_prob(post, predicted);
    if (correct_label == predicted) correct_predictions++;
    total_predictions++;
    cout << "  correct = " << correct_label << ", predicted = "
    << predicted << ", log-probability score = " << log_score
    << "\n" << "  content = " << content << "\n" << endl;
  }
  cout << "performance: " << correct_predictions << " / "
  << total_predictions << " posts predicted correctly" << endl;
}


void Classifier::print_data(const string& filename){
  cout << "training data:" << endl;
  csvstream file(filename);
  map<string, string> line;

  while (file >> line){
    string label = line["tag"];
    string content = line["content"];
    cout << "  label = " << label << ", content = " << content << endl;
  }
  cout << "trained on " << total_posts << " examples" << endl;
  cout << "vocabulary size = " << vocab <<  "\n" << endl;

  cout << "classes:" << endl;
  for (const string &label : labels){
    double log_prior = log(static_cast<double>(posts_per_label.at(label)) / total_posts);
    cout << "  " << label << ", " << posts_per_label.at(label) << " examples, "
    << "log-prior = " << log_prior << endl;
  }

  cout << "classifier parameters:" << endl;
  for (const string &label : labels){
    for (const string &word : all_words){
      bool appears = word_count_per_label.at(label).find(word)
                    != word_count_per_label.at(label).end();
      if (appears){
        int count = word_count_per_label.at(label).at(word);
        double log_likelihood = log(static_cast<double>(count) / 
                                 posts_per_label.at(label));
        cout << "  " << label << ":" << word << ", count = " << count
        << ", log-likelihood = " << log_likelihood << endl;
      }
    }
  }
  cout << endl;
}

double Classifier::calculate_log_prob(const vector<string>& sorted_post, 
                                      const string& label) const {
  double score = log(static_cast<double>(posts_per_label.at(label)) / total_posts);
  for (int i = 0; i < sorted_post.size(); i++){
    string current_word = sorted_post[i];
    auto word_it = word_count_per_label.at(label).find(current_word);

    if (word_it != word_count_per_label.at(label).end()) {
      int word_count = word_it->second;
      score += log(static_cast<double>(word_count) / posts_per_label.at(label));
    } else if (all_words.find(current_word) != all_words.end()) {
      score += log(static_cast<double>(posts_per_word.at(current_word)) / total_posts);
    } else {
      score += log(1.0 / total_posts);
    }
  }
  return score;
}



int main(int argc, char **argv) {
  cout.precision(3);

  if (argc != 2 && argc != 3){
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }
  
  string train_filename = argv[1];

  Classifier classifier;

  try {
    classifier.train(train_filename);
  } catch (const csvstream_exception& e) {
    cout << "Error opening file: " << train_filename << endl;
    return 1;
  }
  
  if (argc == 2){
    classifier.print_data(train_filename);
  }

  if (argc == 3) {
    string test_filename = argv[2];
    try {
      csvstream test(test_filename);
    } catch (const csvstream_exception& e) {
      cout << "Error opening file: " << test_filename << endl;
      return 1;
    }
    classifier.accuracy(test_filename);
  }

  return 0;
}