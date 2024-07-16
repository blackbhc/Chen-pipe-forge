#ifndef SNAPH5IO_HPP
#define SNAPH5IO_HPP
#include "../include/others.hpp"
#include <hdf5.h>
#include <string>
#include <unordered_map>
#include <vector>
using std::string, std::vector, std::unordered_map;

/* // the snapshot formats
enum snapshot_convention { gadget4, arepo, auriga };
// NOTE: at now, only gadget4 is available */

class snapH5io
{
public:
    snapH5io( string sim_data, string analysis_data );
    ~snapH5io();
    void read_datasets( vector< unsigned int >& partNums, vector< double* >& coordinates,
                        vector< double* >& masses, vector< string >& datasetNames );
    void write_rc_infos( unsigned int RbinNum, unsigned int PhiBinNum, double* accs, double* rvs,
                         double* pots, string datasetName );
    void write_coordinates( unsigned int RbinNum, unsigned int PhiBinNum, double Rmin, double Rmax,
                            double* poses );

private:
    hid_t inFile = -1, outFile = -1;  // file id: simulation file and analysis result file
    unordered_map< string, unsigned int > parttype2partNum;
};
#endif  // SNAPH5IO_HPP
