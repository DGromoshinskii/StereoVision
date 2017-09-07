#ifndef EXPORTDLL_H
#define EXPORTDLL_H

#ifdef EXPORT_STEREO_RECONSTRUCTION_DLL

#ifdef _WIN32
#define STEREO_RECONSTRUCTION_DLL __declspec(dllexport)
#elif __GNUC__ >= 4
#define STEREO_RECONSTRUCTION_DLL __attribute__((visibility("default")))
#endif

#else

#ifdef _WIN32
#define STEREO_RECONSTRUCTION_DLL __declspec(dllimport)
#elif __GNUC__ >= 4
#define STEREO_RECONSTRUCTION_DLL __attribute__((visibility("default")))
#endif

#endif

#endif
