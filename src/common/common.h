#ifndef TUYUIDCARD_EXPORT_H_
#define TUYUIDCARD_EXPORT_H_

#define TUYUIDCARD_API_EXPORT
//-----------------------------
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)

#ifdef TUYUIDCARD_API_EXPORT
#define TUYUIDCARD_API __declspec(dllexport)
#else
#define TUYUIDCARD_API __declspec(dllimport)
#endif

#else
// other os
#define TAISHANOCR_API
#endif



#ifdef _WIN32
	#define TUYUIDCARD_HIDDEN
	#if defined(TUYUIDCARD_BUILD_SHARED_LIBS)
		#define TUYUIDCARD_EXPORT __declspec(dllexport)
		#define TUYUIDCARD_IMPORT __declspec(dllimport)
	#else
		#define TUYUIDCARD_EXPORT
		#define TUYUIDCARD_IMPORT
	#endif
#else  // _WIN32
#if defined(__GNUC__)
#define TUYUIDCARD_EXPORT __attribute__((__visibility__("default")))
#define TUYUIDCARD_HIDDEN __attribute__((__visibility__("hidden")))
#else  // defined(__GNUC__)
#define TUYUIDCARD_EXPORT
#define TUYUIDCARD_HIDDEN
#endif  // defined(__GNUC__)
#define TUYUIDCARD_IMPORT TUYUIDCARD_EXPORT
#endif  // _WIN32

#ifdef NO_EXPORT
#undef TUYUIDCARD_EXPORT
#define TUYUIDCARD_EXPORT
#endif


#endif  // TUYUIDCARD_EXPORT_H_