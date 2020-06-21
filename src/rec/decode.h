#ifndef CRNN_DECODE_H_
#define CRNN_DECODE_H_

#include <string>
#include <vector>
extern std::vector<std::string> alphabets;
std::vector<int> GreedyDecode(const std::vector<int> &preds);

#endif //CRNN_DECODE_H_