#ifndef REACTION
#define REACTION

double calcCritDiff(double a, double b);
double f_Q(double a, double b);
double f_R(double a, double b);
double g_Q(double a, double b);
double g_R(double a, double b);
double optFunc(double d, double a, double b);
double jacOptFunc(double d, double a, double b);
double newton(double d, double a, double b);
double calcCritDiff(double a, double b);
double calcCritWavenumber(double a, double b, double gamma);

#endif
