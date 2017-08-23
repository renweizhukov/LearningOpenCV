/*
 * main.cpp
 *
 *  Created on: Jul 9, 2017
 *      Author: renwei
 */

#include <iostream>
#include <string>
#include <algorithm>

#include <boost/program_options.hpp>

#include "VocabularyBuilder.h"
#include "SvmClassifierTrainer.h"
#include "SvmClassifierTester.h"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

/*
 * @function main
 * @brief Main function
 */
int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("command", po::value<string>()->required(), "build | train | test | help")   // This is a positional option.
        ("classifier-prefix,p", po::value<string>(), "The common name prefix (including the directory name) of the files which store the trained classifiers. It is an output for classifier training and an input for classifier testing")
        ("descriptors,e", po::value<string>(), "The yml file which stores the descriptors of all the training images. It is an output for vocabulary building and an input for classifier training")
        ("expected-class,c", po::value<string>(), "The expected class of the test image which will be compared with the class evaluated by the SVM classifiers")
        ("image,i", po::value<string>(), "The image file which will be used for testing")
        ("image-dir,d", po::value<string>(), "The directory of images which will be used for vocabulary building or matcher training or classifier testing")
        ("matcher-file,m", po::value<string>(), "The file which stores the trained FLANN-based matcher. It is an output for matcher training and an input for classifier testing")
        ("result,r", po::value<string>(), "The output file which will store the testing results")
        ("vocabulary,v", po::value<string>(), "The yml file which stores the vocabulary. It is an output for vocabulary building and an input for classifier training and testing");

    po::positional_options_description posOpt;
    posOpt.add("command", 1);   // Only one command is accepted at one execution.

    po::variables_map vm;
    string cmd;

    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).positional(posOpt).run(), vm);
        po::notify(vm);
    }
    catch(po::error& e)
    {
        cerr << "[ERROR]: " << e.what() << endl << endl;
        cout << opt << endl;
        return -1;
    }

    // Get the command and convert the command string into lower case.
    cmd = vm["command"].as<string>();
    transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

    string classifierPrefix;
    string descriptorsFile;
    string expectedClass;
    string imgDir;
    string imgFile;
    string matcherFile;
    string resultFile;
    string vocabularyFile;

    if (cmd == "help")
    {
        cout << opt << endl;
    }
    else if (cmd == "build")
    {
        cout << "[INFO]: Building a vocabulary of the descriptors of all the images" << endl;

        if (vm.count("descriptors") == 0)
        {
            cerr << "[ERROR]: A yml file is required to be given for storing the descriptors." << endl << endl;
            return -1;
        }

        if (vm.count("image-dir") == 0)
        {
            cerr << "[ERROR]: An image base path is required to be given for building the vocabulary." << endl << endl;
            return -1;
        }

        if (vm.count("vocabulary") == 0)
        {
            cerr << "[ERROR]: A vocabulary yml file is required to be given for storing the vocabulary result." << endl << endl;
            return -1;
        }

        descriptorsFile = vm["descriptors"].as<string>();
        imgDir = vm["image-dir"].as<string>();
        vocabularyFile = vm["vocabulary"].as<string>();
        VocabularyBuilder builder(imgDir, descriptorsFile, vocabularyFile);

        // We don't need the output descriptors and vocabulary, so we pass noArray() here.
        builder.ComputeDescriptors(noArray());
        builder.BuildVocabulary(noArray());
    }
    else if (cmd == "train")
    {
        cout << "[INFO]: Training the SVM classifiers" << endl;

        if (vm.count("classifier-prefix") == 0)
        {
            cerr << "[ERROR]: A common filename prefix (including the directory name) is required to be given for storing the trained classifiers." << endl << endl;
            return -1;
        }

        if (vm.count("descriptors") == 0)
        {
            cerr << "[ERROR]: A yml file is required to be given for loading the descriptors." << endl << endl;
            return -1;
        }

        if (vm.count("image-dir") == 0)
        {
            cerr << "[ERROR]: An image base path is required to be given for training the FLANN-based matcher." << endl << endl;
            return -1;
        }

        if (vm.count("matcher-file") == 0)
        {
            cerr << "[ERROR]: A yml file is required to be given for storing the trained FLANN-based matcher." << endl << endl;
            return -1;
        }

        if (vm.count("vocabulary") == 0)
        {
            cerr << "[ERROR]: A yml file is required to be given for loading the vocabulary." << endl << endl;
            return -1;
        }

        classifierPrefix = vm["classifier-prefix"].as<string>();
        descriptorsFile = vm["descriptors"].as<string>();
        imgDir = vm["image-dir"].as<string>();
        matcherFile = vm["matcher-file"].as<string>();
        vocabularyFile = vm["vocabulary"].as<string>();

        SvmClassifierTrainer svmTrainer(vocabularyFile, descriptorsFile, imgDir, matcherFile, classifierPrefix);
        svmTrainer.Train();
    }
    else if (cmd == "test")
    {
        cout << "[INFO]: Testing the SVM classifiers" << endl;

        if (vm.count("classifier-prefix") == 0)
        {
            cerr << "[ERROR]: A common filename prefix (including the directory name) is required to be given for loading the trained classifiers." << endl << endl;
            return -1;
        }

        if ((vm.count("image") > 0) && (vm.count("expected-class") == 0))
        {
            cerr << "[ERROR]: The expected class of the test image is required to be compared with the evaluated class." << endl << endl;
            return -1;
        }

        if ((vm.count("image") == 0) && (vm.count("image-dir") == 0))
        {
            cerr << "[ERROR]: Either an image or an image base path is required to be given for testing the classifiers." << endl << endl;
            return -1;
        }

        if (vm.count("matcher-file") == 0)
        {
            cerr << "[ERROR]: A matcher file is required to be given for loading the trained FLANN-based matcher." << endl << endl;
            return -1;
        }

        if (vm.count("result") == 0)
        {
            cerr << "[ERROR]: A file is required to be given for recording the testing result." << endl << endl;
            return -1;
        }

        if (vm.count("vocabulary") == 0)
        {
            cerr << "[ERROR]: A yml file is required to be given for loading the vocabulary." << endl << endl;
            return -1;
        }

        classifierPrefix = vm["classifier-prefix"].as<string>();
        matcherFile = vm["matcher-file"].as<string>();
        resultFile = vm["result"].as<string>();
        vocabularyFile = vm["vocabulary"].as<string>();

        SvmClassifierTester svmTester(vocabularyFile, classifierPrefix, matcherFile, resultFile);
        if (!svmTester.InitBowImgDescriptorExtractor())
        {
            cerr << "[ERROR]: Failed to initialize the BOWImgDescriptorExtractor from the vocabulary file " << vocabularyFile << "." << endl << endl;
            return -1;
        }

        if (!svmTester.InitSvmClassifiers())
        {
            cerr << "[ERROR]: Failed to initialize the SVM classifiers from the classifier prefix " << classifierPrefix << "." << endl << endl;
            return -1;
        }

        if (!svmTester.InitFlannBasedMatcher())
        {
            cerr << "[ERROR]: Failed to initialize the FLANN-based matcher from the matcher file " << matcherFile << "." << endl << endl;
            return -1;
        }

        if (vm.count("image") > 0)
        {
            expectedClass = vm["expected-class"].as<string>();
            imgFile = vm["image"].as<string>();
            svmTester.EvaluateOneImg(imgFile, expectedClass);
        }
        else
        {
            // i.e., vm.count("image-dir") > 0
            imgDir = vm["image-dir"].as<string>();
            svmTester.EvaluateImgs(imgDir);
        }
    }
    else
    {
        cerr << "[ERROR]: " << "Unknown command " << cmd << endl << endl;
        cout << opt << endl;
        return -1;
    }

    return 0;
}
