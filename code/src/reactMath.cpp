#include <iostream>
#include <math.h>
#include <ostream>
#include <reactMath.hpp>

double f_Q(double a, double b){
    return (b-a)/(a+b);
}

double f_R(double a, double b){
    return pow(a+b, 2);
}

double g_Q(double a, double b){
    return -2 * b / (a + b);
}

double g_R(double a, double b){
    return -1. * f_R(a, b);
}

double optFunc(double d, double a, double b){
    double temp = 0;

    temp += pow(d * f_Q(a,b), 2);
    temp += 2 * (2 * f_R(a,b) * g_Q(a,b) - f_Q(a,b) * g_R(a,b)) * d;
    temp += g_R(a,b);
    return temp; 
}

double jacOptFunc(double d, double a, double b){
    
    double temp = 0;

    temp += 2 * d * pow(f_Q(a,b),2);
    temp += 2 * (2 * f_R(a,b) * g_Q(a,b) - f_Q(a,b) * g_R(a,b));

    return temp;
}

double newton(double x, double a, double b){
    for(auto i = 0; i < 50; i++){
        x = x - optFunc(x, a, b) / jacOptFunc(x, a, b);
        if(optFunc(x,a,b) < 1e-12){
            std::cout   << "Converged:" << std::endl
                        << "    " << optFunc(x,a,b) << std::endl
                        << "    " << x << std::endl;
            return x;
        }
    }

    std::cout   << "Failed to converge:" << std::endl
                << "    " << optFunc(x,a,b) << std::endl
                << "    " << x << std::endl;
    return x;
}

double calcCritDiff(double a, double b){
    
    bool tc1 = (f_Q(a,b) + g_R(a,b) < 0);
    bool tc2 = (f_Q(a,b) * g_R(a,b) - f_R(a,b) * g_Q(a,b) > 0);
    bool tc3 = (f_Q(a,b) * g_R(a,b) < 0);

    if (!(tc1 && tc2 && tc3)){
        std::cout   << "Warning, Turing conditions imply no "
                    << "critical diffusion exists."
                    << std::endl;
        return 1.;
    };

    return newton(40., a, b);

}

double calcCritWavenumber(double a, double b, double gamma){
    double d = calcCritDiff(a, b);

    return std::sqrt(gamma * (d * f_Q(a,b) + g_R(a,b)) / (2 * d));
}
