%module mnn
%{
// Inference
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"
// Express
#include "MNN_generated.h"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/ExecutorScope.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/Module.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN/expr/Optimizer.hpp"
#include "MNN/expr/Scope.hpp"
// Image Processing
#include "MNN/ImageProcess.hpp"
#include "MNN/Rect.h"
#include "MNN/Matrix.h"
// Training
#include "Dataset.hpp"
#include "DataLoader.hpp"
#include "DataLoaderConfig.hpp"
#include "Example.hpp"
#include "Sampler.hpp"
#include "Transform.hpp"
#include "ImageDataset.hpp"
#include "MnistDataset.hpp"
//#include "Lenet.hpp"
//#include "MobilenetV1.hpp"
//#include "MobilenetV2.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#include "ADAM.hpp"
#include "Loss.hpp"
#include "Transformer.hpp"
%}

#define __attribute__(x)  // to remove it in swig
#define final // to remove it in swig

%include <std_vector.i>
%include <std_list.i>
%include <std_map.i>
%include <std_pair.i>
%include <std_string.i>
%include <std_shared_ptr.i>
%include <exception.i>
%include <typemaps.i>

%template(S8Vec) std::vector<char>;
%template(U8Vec) std::vector<unsigned char>;
%template(S32Vec) std::vector<int>;
%template(U32Vec) std::vector<unsigned int>;
%template(F32Vec) std::vector<float>;
%template(F64Vec) std::vector<double>;
%template(StrVec) std::vector<std::string>;
//%template(PairS32S32) std::pair<unsigned int, unsigned int>;
//%template(VecPairS32S32) std::vector<std::pair<unsigned int, unsigned int>>;
//%template(VecVecPairS32S32) std::vector< std::vector< std::pair<unsigned int, unsigned int> > >;

%template(ScheduleConfigVec) std::vector<MNN::ScheduleConfig>;
//%template(MNNForwardTypeVec) std::vector<MNN::MNNForwardType>;
%template(TensorMap) std::map<std::string, MNN::Tensor*>;
%template(TensorPtrVec) std::vector<MNN::Tensor*>;
%template(SessionPtrVec) std::vector<MNN::Session*>;
%template(VarpVec) std::vector<MNN::Express::VARP>;
//%template(ExprpVec) std::vector<MNN::Express::EXPRP>;
//%template(ExampleVec) std::vector<MNN::Train::Example>;

%shared_ptr(MNN::Backend)
%shared_ptr(MNN::Runtime)
%shared_ptr(MNN::BufferStorage)
%shared_ptr(MNN::Express::Executor)
%shared_ptr(MNN::Express::Executor::ComputeCache)
%shared_ptr(MNN::Express::Executor::RuntimeManager)
%shared_ptr(MNN::Express::Expr)
%shared_ptr(MNN::Express::Expr::Inside)
%shared_ptr(MNN::Express::Module)
%shared_ptr(MNN::Express::Optimizer)
%shared_ptr(MNN::Express::Optimizer::Parameters)
%shared_ptr(MNN::Express::Variable)
//%unique_ptr(MNN::OpT)

// This method is renamed to a valid Python method name, as otherwise
// it cannot be used from Python
%rename(inc)   *::operator++;
%rename(dec)   *::operator--;
%ignore *::operator=;
%ignore *::operator[];

%ignore MNN::Train::InferOptimizer;
%ignore MNN::Train::TurnTrainable;

%rename(add) MNN::Express::VARP::operator+;
%rename(sub) MNN::Express::VARP::operator-;
%rename(mul) MNN::Express::VARP::operator*;
%rename(div) MNN::Express::VARP::operator/;
%rename(equ) MNN::Express::VARP::operator==;
%rename(nequ) MNN::Express::VARP::operator!=;
%rename(less) MNN::Express::VARP::operator<;
%rename(le) MNN::Express::VARP::operator<=;
%rename(reset) MNN::Express::VARP::operator=;

%include "MNN/HalideRuntime.h"
%template(Halide_Type_Float_func) halide_type_of<float>;
%template(Halide_Type_Int8_func) halide_type_of<int8_t>;
%template(Halide_Type_Uint8_func) halide_type_of<uint8_t>;

%include "MNN/MNNDefine.h"
%include "MNN/MNNForwardType.h"
// Inference
%include "MNN/ErrorCode.hpp"
%include "MNN/Tensor.hpp"
%include "MNN/Interpreter.hpp"
// Express
%include "MNN/expr/Expr.hpp"
%include "MNN/expr/Executor.hpp"
%include "MNN/expr/ExecutorScope.hpp"
%include "MNN/expr/ExprCreator.hpp"
%include "MNN/expr/MathOp.hpp"
%include "MNN/expr/Module.hpp"
%include "MNN/expr/NeuralNetWorkOp.hpp"
%include "MNN/expr/Optimizer.hpp"
%include "MNN/expr/Scope.hpp"
// Image Processing
%include "MNN/ImageProcess.hpp"
//%include "MNN/Matrix.h"
//%include "MNN/Rect.h"
// Training
%include "Example.hpp"
%include "Sampler.hpp"
%include "DataLoaderConfig.hpp"
%include "ParameterOptimizer.hpp"
%include "Dataset.hpp"
%include "DataLoader.hpp"
%include "Transform.hpp"
%include "ImageDataset.hpp"
%include "MnistDataset.hpp"
//%include "Lenet.hpp"
//%include "MobilenetV1.hpp"
//%include "MobilenetV2.hpp"
%include "NN.hpp"
%include "SGD.hpp"
%include "ADAM.hpp"
%include "Loss.hpp"
%include "Transformer.hpp"