#include <bits/stdc++.h>
using namespace std;


// Function to generate a random DNA sequence of given length
std::string generateRandomDNASequence(int length) {
    // Define the DNA alphabet
    const std::string dnaAlphabet = "ACGT";
    std::string sequence = "";

    // Seed the random number generator
    std::srand(std::time(0));

    // Generate the random sequence
    for (int i = 0; i < length; ++i) {
        // Generate a random index to choose a character from the DNA alphabet
        int randomIndex = std::rand() % dnaAlphabet.length();
        // Append the randomly chosen character to the sequence
        sequence += dnaAlphabet[randomIndex];
    }

    return sequence;
}


string insertMotifMiddle(string input, const string& motif) {
    int motifLen = motif.size();
    int inputLen = input.size();

    int limit = inputLen - motifLen - 5;
     int start = inputLen/2 - motifLen/2;
//    int start = rand() % limit;

    for(int i = 0; i < motifLen; i++) {
        int j = i + start;
        input[j] = motif[i];
    }
    return input;
}

string insertMotifStartOrEnd(string input, const string& motif) {
    int motifLen = motif.size();
    int inputLen = input.size();

    int limit = inputLen - motifLen - 5;
    // int start = inputLen/2 - motifLen/2;
    int r = rand() % 2;
    int start = inputLen / 3;
    if(r == 1) {
        start = start * 2;
    }

    for(int i = 0; i < motifLen; i++) {
        int j = i + start;
        input[j] = motif[i];
    }
    return input;
}

string insertMotifRandom(string input, const string& motif) {
    int motifLen = motif.size();
    int inputLen = input.size();

    int limit = inputLen - motifLen - 5;
    // int start = inputLen/2 - motifLen/2;
    int start = rand() % limit;

    for(int i = 0; i < motifLen; i++) {
        int j = i + start;
        input[j] = motif[i];
    }
    return input;
}

string insertMotif(string input, const string& motif) {
//    return insertMotifMiddle(input, motif);
    return insertMotifStartOrEnd(input, motif);
//    return insertMotifRandom(input, motif);
}

void start() {
 string motif = "CTCATGTCA";
    int window = 500;

    cout<<"Sequence,class\n";
    for(int i=0; i<1000; i++) {
        string mSeq = generateRandomDNASequence(window);
        int j = i % 2;
        if(j) {
            mSeq = insertMotif(mSeq, motif);
        }
        cout<<mSeq <<", "<< j <<".0\n";
    }
//    return 0;
}

int main() {
 string motif = "CTCATGTCA";
    cout<<"Sequence,class,\n";
   for(int i=0; i<1000; i++) {
    string mSeq;
    cin>>mSeq;
    int j = i % 2;
    if(j) {
        mSeq = insertMotif(mSeq, motif);
    }
    cout<<mSeq <<", "<< j <<".0,\n";
   }

   return 0;
}