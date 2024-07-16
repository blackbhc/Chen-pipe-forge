#ifndef RC_ANA_HPP
#define RC_ANA_HPP
#include "../include/snapH5io.hpp"
#include "others.hpp"
#include <string>
using std::string;
class analyzer_of_snapshot
{
private:
    unsigned int            pos_num = 0, rBinNum = 0, phiBinNum = 0;
    double                  rMin = 0, rMax = 0;
    dynamic_array< double > coordinates;
    snapH5io                h5io;


public:
    analyzer_of_snapshot( double Rmin, double Rmax, unsigned int RbinNum, unsigned int PhiBinNum,
                          string in_filename, string out_filename );
    void read_ana_write();
};

#endif
