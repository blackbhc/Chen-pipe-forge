#include "../include/others.hpp"
#include <cmath>
#include <numeric>

vec3::vec3( double data[ 3 ] )
{
    for ( int i = 0; i < 3; ++i )
    {
        this->data[ i ] = data[ i ];
    }
}

double vec3::norm()
{
    double sumSquare = 0;
    for ( int i = 0; i < 3; ++i )
        sumSquare += this->data[ i ] * this->data[ i ];
    return std::pow( sumSquare, 0.5 );
}

void vec3::normalize()
{
    double selfNorm = this->norm();
    for ( int i = 0; i < 3; ++i )
        this->data[ i ] /= selfNorm;
}

double vec3::dot( vec3& another )
{
    double sum = 0;
    for ( int i = 0; i < 3; ++i )
        sum += this->data[ i ] * another.data[ i ];
    return sum;
}

vec3 vec3::cross( vec3& another )
{
    double tmp[ 3 ] = { 0, 0, 0 };
    vec3   result( tmp );
    result.data[ 0 ] = this->data[ 1 ] * another.data[ 2 ] - this->data[ 2 ] * another.data[ 1 ];
    result.data[ 1 ] = this->data[ 2 ] * another.data[ 0 ] - this->data[ 0 ] * another.data[ 2 ];
    result.data[ 2 ] = this->data[ 0 ] * another.data[ 1 ] - this->data[ 1 ] * another.data[ 0 ];
    return result;
}

void vec3::assign_to( double* array_ptr )
{
    for ( int i = 0; i < 3; ++i )
        *( array_ptr + i ) = this->data[ i ];
}
