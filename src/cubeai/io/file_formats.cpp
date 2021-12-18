#include "cubeai/io/file_formats.h"

namespace cubeai
{

std::string
FileFormats::type_to_string(FileFormats::Type t){

   switch(t)
   {
      case FileFormats::Type::CSV:
         return "csv";
     case FileFormats::Type::JSON:
        return "json";
   }

   return "INVALID_TYPE";
}

}
