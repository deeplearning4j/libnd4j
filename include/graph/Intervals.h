//
// Created by yurii@skymind.io on 24.10.2017.
//

#ifndef LIBND4J_INTERVALS_H
#define LIBND4J_INTERVALS_H

#include <vector>
#include <initializer_list>


namespace  nd4j {

    class Intervals {
    
    private:
        std::initializer_list<std::vector<int>> _content;

    public:

        Intervals() { }

        // default constructor
        Intervals();
        
        // constructor
        Intervals(const std::initializer_list<std::vector<int>>& content );
        
        // accessing operator
        std::vector<int> operator[](const int i) const;

        // returns size of _content
        int size() const;

    };


}

#endif //LIBND4J_INTERVALS_H
